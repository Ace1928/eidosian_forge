from __future__ import annotations
import typing as t
from datetime import datetime
from markupsafe import escape
from markupsafe import Markup
from ._internal import _get_environ
class FailedDependency(HTTPException):
    """*424* `Failed Dependency`

    Used if the method could not be performed on the resource
    because the requested action depended on another action and that action failed.
    """
    code = 424
    description = 'The method could not be performed on the resource because the requested action depended on another action and that action failed.'