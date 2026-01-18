from __future__ import annotations
import abc
import datetime
import enum
from collections.abc import MutableMapping as _MutableMapping
from typing import (
from bson.binary import (
from bson.typings import _DocumentType
def _options_dict(self) -> dict[str, Any]:
    """Dictionary of the arguments used to create this object."""
    return {'document_class': self.document_class, 'tz_aware': self.tz_aware, 'uuid_representation': self.uuid_representation, 'unicode_decode_error_handler': self.unicode_decode_error_handler, 'tzinfo': self.tzinfo, 'type_registry': self.type_registry, 'datetime_conversion': self.datetime_conversion}