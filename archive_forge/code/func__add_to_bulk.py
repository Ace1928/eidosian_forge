from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
def _add_to_bulk(self, bulkobj: _Bulk) -> None:
    """Add this operation to the _Bulk instance `bulkobj`."""
    bulkobj.add_update(self._filter, self._doc, True, self._upsert, collation=validate_collation_or_none(self._collation), array_filters=self._array_filters, hint=self._hint)