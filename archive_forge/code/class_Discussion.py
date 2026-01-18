from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@dataclass
class Discussion:
    """
    A Discussion or Pull Request on the Hub.

    This dataclass is not intended to be instantiated directly.

    Attributes:
        title (`str`):
            The title of the Discussion / Pull Request
        status (`str`):
            The status of the Discussion / Pull Request.
            It must be one of:
                * `"open"`
                * `"closed"`
                * `"merged"` (only for Pull Requests )
                * `"draft"` (only for Pull Requests )
        num (`int`):
            The number of the Discussion / Pull Request.
        repo_id (`str`):
            The id (`"{namespace}/{repo_name}"`) of the repo on which
            the Discussion / Pull Request was open.
        repo_type (`str`):
            The type of the repo on which the Discussion / Pull Request was open.
            Possible values are: `"model"`, `"dataset"`, `"space"`.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        is_pull_request (`bool`):
            Whether or not this is a Pull Request.
        created_at (`datetime`):
            The `datetime` of creation of the Discussion / Pull Request.
        endpoint (`str`):
            Endpoint of the Hub. Default is https://huggingface.co.
        git_reference (`str`, *optional*):
            (property) Git reference to which changes can be pushed if this is a Pull Request, `None` otherwise.
        url (`str`):
            (property) URL of the discussion on the Hub.
    """
    title: str
    status: DiscussionStatus
    num: int
    repo_id: str
    repo_type: str
    author: str
    is_pull_request: bool
    created_at: datetime
    endpoint: str

    @property
    def git_reference(self) -> Optional[str]:
        """
        If this is a Pull Request , returns the git reference to which changes can be pushed.
        Returns `None` otherwise.
        """
        if self.is_pull_request:
            return f'refs/pr/{self.num}'
        return None

    @property
    def url(self) -> str:
        """Returns the URL of the discussion on the Hub."""
        if self.repo_type is None or self.repo_type == REPO_TYPE_MODEL:
            return f'{self.endpoint}/{self.repo_id}/discussions/{self.num}'
        return f'{self.endpoint}/{self.repo_type}s/{self.repo_id}/discussions/{self.num}'