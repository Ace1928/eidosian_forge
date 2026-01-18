from typing import (
import attr
from .parsing import (
@attr.s(auto_attribs=True)
class PostcommandData:
    """Data class containing information passed to postcommand hook methods"""
    stop: bool
    statement: Statement