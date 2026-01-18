from typing import (
import attr
from .parsing import (
@attr.s(auto_attribs=True)
class PostparsingData:
    """Data class containing information passed to postparsing hook methods"""
    stop: bool
    statement: Statement