import base64
from abc import ABC
from datetime import datetime
from typing import Callable, Dict, Iterator, List, Literal, Optional, Union
import requests
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
class GitHubIssuesLoader(BaseGitHubLoader):
    """Load issues of a GitHub repository."""
    include_prs: bool = True
    'If True include Pull Requests in results, otherwise ignore them.'
    milestone: Union[int, Literal['*', 'none'], None] = None
    "If integer is passed, it should be a milestone's number field.\n        If the string '*' is passed, issues with any milestone are accepted.\n        If the string 'none' is passed, issues without milestones are returned.\n    "
    state: Optional[Literal['open', 'closed', 'all']] = None
    "Filter on issue state. Can be one of: 'open', 'closed', 'all'."
    assignee: Optional[str] = None
    "Filter on assigned user. Pass 'none' for no user and '*' for any user."
    creator: Optional[str] = None
    'Filter on the user that created the issue.'
    mentioned: Optional[str] = None
    "Filter on a user that's mentioned in the issue."
    labels: Optional[List[str]] = None
    'Label names to filter one. Example: bug,ui,@high.'
    sort: Optional[Literal['created', 'updated', 'comments']] = None
    "What to sort results by. Can be one of: 'created', 'updated', 'comments'.\n        Default is 'created'."
    direction: Optional[Literal['asc', 'desc']] = None
    "The direction to sort the results by. Can be one of: 'asc', 'desc'."
    since: Optional[str] = None
    'Only show notifications updated after the given time.\n        This is a timestamp in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.'
    page: Optional[int] = None
    'The page number for paginated results. \n        Defaults to 1 in the GitHub API.'
    per_page: Optional[int] = None
    'Number of items per page. \n        Defaults to 30 in the GitHub API.'

    @validator('since', allow_reuse=True)
    def validate_since(cls, v: Optional[str]) -> Optional[str]:
        if v:
            try:
                datetime.strptime(v, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                raise ValueError(f"Invalid value for 'since'. Expected a date string in YYYY-MM-DDTHH:MM:SSZ format. Received: {v}")
        return v

    def lazy_load(self) -> Iterator[Document]:
        """
        Get issues of a GitHub repository.

        Returns:
            A list of Documents with attributes:
                - page_content
                - metadata
                    - url
                    - title
                    - creator
                    - created_at
                    - last_update_time
                    - closed_time
                    - number of comments
                    - state
                    - labels
                    - assignee
                    - assignees
                    - milestone
                    - locked
                    - number
                    - is_pull_request
        """
        url: Optional[str] = self.url
        while url:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            issues = response.json()
            for issue in issues:
                doc = self.parse_issue(issue)
                if not self.include_prs and doc.metadata['is_pull_request']:
                    continue
                yield doc
            if response.links and response.links.get('next') and (not self.page and (not self.per_page)):
                url = response.links['next']['url']
            else:
                url = None

    def parse_issue(self, issue: dict) -> Document:
        """Create Document objects from a list of GitHub issues."""
        metadata = {'url': issue['html_url'], 'title': issue['title'], 'creator': issue['user']['login'], 'created_at': issue['created_at'], 'comments': issue['comments'], 'state': issue['state'], 'labels': [label['name'] for label in issue['labels']], 'assignee': issue['assignee']['login'] if issue['assignee'] else None, 'milestone': issue['milestone']['title'] if issue['milestone'] else None, 'locked': issue['locked'], 'number': issue['number'], 'is_pull_request': 'pull_request' in issue}
        content = issue['body'] if issue['body'] is not None else ''
        return Document(page_content=content, metadata=metadata)

    @property
    def query_params(self) -> str:
        """Create query parameters for GitHub API."""
        labels = ','.join(self.labels) if self.labels else self.labels
        query_params_dict = {'milestone': self.milestone, 'state': self.state, 'assignee': self.assignee, 'creator': self.creator, 'mentioned': self.mentioned, 'labels': labels, 'sort': self.sort, 'direction': self.direction, 'since': self.since, 'page': self.page, 'per_page': self.per_page}
        query_params_list = [f'{k}={v}' for k, v in query_params_dict.items() if v is not None]
        query_params = '&'.join(query_params_list)
        return query_params

    @property
    def url(self) -> str:
        """Create URL for GitHub API."""
        return f'{self.github_api_url}/repos/{self.repo}/issues?{self.query_params}'