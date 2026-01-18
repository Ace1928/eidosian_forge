from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@dataclass
class DiscussionComment(DiscussionEvent):
    """A comment in a Discussion / Pull Request.

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
        content (`str`):
            The raw markdown content of the comment. Mentions, links and images are not rendered.
        edited (`bool`):
            Whether or not this comment has been edited.
        hidden (`bool`):
            Whether or not this comment has been hidden.
    """
    content: str
    edited: bool
    hidden: bool

    @property
    def rendered(self) -> str:
        """The rendered comment, as a HTML string"""
        return self._event['data']['latest']['html']

    @property
    def last_edited_at(self) -> datetime:
        """The last edit time, as a `datetime` object."""
        return parse_datetime(self._event['data']['latest']['updatedAt'])

    @property
    def last_edited_by(self) -> str:
        """The last edit time, as a `datetime` object."""
        return self._event['data']['latest'].get('author', {}).get('name', 'deleted')

    @property
    def edit_history(self) -> List[dict]:
        """The edit history of the comment"""
        return self._event['data']['history']

    @property
    def number_of_edits(self) -> int:
        return len(self.edit_history)