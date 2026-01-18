from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, cast
from bson.codec_options import _parse_codec_options
from pymongo import common
from pymongo.auth import MongoCredential, _build_credentials_tuple
from pymongo.common import validate_boolean
from pymongo.compression_support import CompressionSettings
from pymongo.errors import ConfigurationError
from pymongo.monitoring import _EventListener, _EventListeners
from pymongo.pool import PoolOptions
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import (
from pymongo.server_selectors import any_server_selector
from pymongo.ssl_support import get_ssl_context
from pymongo.write_concern import WriteConcern
def _parse_read_concern(options: Mapping[str, Any]) -> ReadConcern:
    """Parse read concern options."""
    concern = options.get('readconcernlevel')
    return ReadConcern(concern)