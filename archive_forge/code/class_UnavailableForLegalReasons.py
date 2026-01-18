from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class UnavailableForLegalReasons(HTTPException):
    """*451* `Unavailable For Legal Reasons`

    This status code indicates that the server is denying access to the
    resource as a consequence of a legal demand.
    """
    code = 451
    description = 'Unavailable for legal reasons.'