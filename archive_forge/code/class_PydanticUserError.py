from __future__ import annotations as _annotations
import re
from typing_extensions import Literal, Self
from ._migration import getattr_migration
from .version import version_short
class PydanticUserError(PydanticErrorMixin, TypeError):
    """An error raised due to incorrect use of Pydantic."""