from inspect import iscoroutine, isgenerator
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from warnings import warn
import attr
@attr.s(hash=False, eq=False, repr=False, auto_attribs=True)
class CDATA:
    """
    A C{<![CDATA[]]>} block from a template.  Given a separate representation in
    the DOM so that they may be round-tripped through rendering without losing
    information.
    """
    data: str
    'The data between "C{<![CDATA[}" and "C{]]>}".'

    def __repr__(self) -> str:
        return f'CDATA({self.data!r})'