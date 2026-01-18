from __future__ import annotations
import collections.abc as collections_abc
import re
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union
from urllib.parse import parse_qsl
from urllib.parse import quote
from urllib.parse import quote_plus
from urllib.parse import unquote
from .interfaces import Dialect
from .. import exc
from .. import util
from ..dialects import plugins
from ..dialects import registry
@util.deprecated('1.4', 'The :meth:`_engine.URL.__to_string__ method is deprecated and will be removed in a future release.  Please use the :meth:`_engine.URL.render_as_string` method.')
def __to_string__(self, hide_password: bool=True) -> str:
    """Render this :class:`_engine.URL` object as a string.

        :param hide_password: Defaults to True.   The password is not shown
         in the string unless this is set to False.

        """
    return self.render_as_string(hide_password=hide_password)