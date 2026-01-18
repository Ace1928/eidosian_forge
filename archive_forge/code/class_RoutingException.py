from __future__ import annotations
import difflib
import typing as t
from ..exceptions import BadRequest
from ..exceptions import HTTPException
from ..utils import cached_property
from ..utils import redirect
class RoutingException(Exception):
    """Special exceptions that require the application to redirect, notifying
    about missing urls, etc.

    :internal:
    """