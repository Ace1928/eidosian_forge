from hamcrest.core.selfdescribing import SelfDescribing
from .base_description import BaseDescription
class StringDescription(BaseDescription):
    """A :py:class:`~hamcrest.core.description.Description` that is stored as a
    string.

    """

    def __init__(self) -> None:
        self.out = ''

    def __str__(self) -> str:
        """Returns the description."""
        return self.out

    def append(self, string: str) -> None:
        self.out += str(string)