from __future__ import annotations
import sys
import pydantic
from ._utils import Colors, organization_info
from .._exceptions import APIError, OpenAIError
class SilentCLIError(CLIError):
    ...