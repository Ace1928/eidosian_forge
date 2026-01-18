from typing import ClassVar, List, Sequence
from twisted.python.util import FancyEqMixin
def _withAttribute(self, name, value):
    if getattr(self, name) != value:
        attr = self.copy()
        attr._subtracting = not value
        setattr(attr, name, value)
        return attr
    else:
        return self.copy()