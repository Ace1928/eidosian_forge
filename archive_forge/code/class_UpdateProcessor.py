from __future__ import annotations
import itertools
import types
import warnings
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
class UpdateProcessor:
    """
    Update plugin interface.

    This module is useful for those wanting to write an update
    processor that can plugin to rdflib. If you are wanting to execute
    an update statement you likely want to do so through the Graph
    class update method.

    .. versionadded:: 4.0

    """

    def __init__(self, graph: 'Graph'):
        pass

    def update(self, strOrQuery: Union[str, 'Update'], initBindings: Mapping['str', 'Identifier']={}, initNs: Mapping[str, Any]={}) -> None:
        pass