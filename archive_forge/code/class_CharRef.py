from inspect import iscoroutine, isgenerator
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from warnings import warn
import attr
@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class CharRef:
    """
    A numeric character reference.  Given a separate representation in the DOM
    so that non-ASCII characters may be output as pure ASCII.

    @since: 12.0
    """
    ordinal: int
    'The ordinal value of the unicode character to which this object refers.'

    def __repr__(self) -> str:
        return 'CharRef(%d)' % (self.ordinal,)