from __future__ import annotations
import typing as t
from ...constants import (
from ...completion import (
from ...util import (
from ...host_configs import (
from ...become import (
from ..argparsing.parsers import (
from .value_parsers import (
from .helpers import (
class EmptyKeyValueParser(KeyValueParser):
    """Composite argument parser when a key/value parser is required but there are no keys available."""

    def get_parsers(self, state: ParserState) -> dict[str, Parser]:
        """Return a dictionary of key names and value parsers."""
        return {}