from typing import (
import attr
from .parsing import (
@attr.s(auto_attribs=True)
class PrecommandData:
    """Data class containing information passed to precommand hook methods"""
    statement: Statement