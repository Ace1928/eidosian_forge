from typing import cast
from zope.interface import Attribute, Interface, implementer
from twisted.web import sux
def addContent(self, text: str) -> str:
    """Add some text data to this Element."""
    if not isinstance(text, str):
        raise TypeError(f'Expected str not {text!r} ({type(text).__name__})')
    c = self.children
    if len(c) > 0 and isinstance(c[-1], str):
        c[-1] = c[-1] + text
    else:
        c.append(text)
    return cast(str, c[-1])