from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@dataclass
class DiscussionWithDetails(Discussion):
    """
    Subclass of [`Discussion`].

    Attributes:
        title (`str`):
            The title of the Discussion / Pull Request
        status (`str`):
            The status of the Discussion / Pull Request.
            It can be one of:
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
        events (`list` of [`DiscussionEvent`])
            The list of [`DiscussionEvents`] in this Discussion or Pull Request.
        conflicting_files (`Union[List[str], bool, None]`, *optional*):
            A list of conflicting files if this is a Pull Request.
            `None` if `self.is_pull_request` is `False`.
            `True` if there are conflicting files but the list can't be retrieved.
        target_branch (`str`, *optional*):
            The branch into which changes are to be merged if this is a
            Pull Request . `None`  if `self.is_pull_request` is `False`.
        merge_commit_oid (`str`, *optional*):
            If this is a merged Pull Request , this is set to the OID / SHA of
            the merge commit, `None` otherwise.
        diff (`str`, *optional*):
            The git diff if this is a Pull Request , `None` otherwise.
        endpoint (`str`):
            Endpoint of the Hub. Default is https://huggingface.co.
        git_reference (`str`, *optional*):
            (property) Git reference to which changes can be pushed if this is a Pull Request, `None` otherwise.
        url (`str`):
            (property) URL of the discussion on the Hub.
    """
    events: List['DiscussionEvent']
    conflicting_files: Union[List[str], bool, None]
    target_branch: Optional[str]
    merge_commit_oid: Optional[str]
    diff: Optional[str]