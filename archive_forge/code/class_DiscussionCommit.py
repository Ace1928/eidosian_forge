from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@dataclass
class DiscussionCommit(DiscussionEvent):
    """A commit in a Pull Request.

    Subclass of [`DiscussionEvent`].

    Attributes:
        id (`str`):
            The ID of the event. An hexadecimal string.
        type (`str`):
            The type of the event.
        created_at (`datetime`):
            A [`datetime`](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)
            object holding the creation timestamp for the event.
        author (`str`):
            The username of the Discussion / Pull Request author.
            Can be `"deleted"` if the user has been deleted since.
        summary (`str`):
            The summary of the commit.
        oid (`str`):
            The OID / SHA of the commit, as a hexadecimal string.
    """
    summary: str
    oid: str