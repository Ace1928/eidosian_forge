from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class UpdateOne(_UpdateOp):
    """Represents an update_one operation."""
    __slots__ = ()

    def __init__(self, filter: Mapping[str, Any], update: Union[Mapping[str, Any], _Pipeline], upsert: bool=False, collation: Optional[_CollationIn]=None, array_filters: Optional[list[Mapping[str, Any]]]=None, hint: Optional[_IndexKeyHint]=None) -> None:
        """Represents an update_one operation.

        For use with :meth:`~pymongo.collection.Collection.bulk_write`.

        :Parameters:
          - `filter`: A query that matches the document to update.
          - `update`: The modifications to apply.
          - `upsert` (optional): If ``True``, perform an insert if no documents
            match the filter.
          - `collation` (optional): An instance of
            :class:`~pymongo.collation.Collation`.
          - `array_filters` (optional): A list of filters specifying which
            array elements an update should apply.
          - `hint` (optional): An index to use to support the query
            predicate specified either by its string name, or in the same
            format as passed to
            :meth:`~pymongo.collection.Collection.create_index` (e.g.
            ``[('field', ASCENDING)]``). This option is only supported on
            MongoDB 4.2 and above.

        .. versionchanged:: 3.11
           Added the `hint` option.
        .. versionchanged:: 3.9
           Added the ability to accept a pipeline as the `update`.
        .. versionchanged:: 3.6
           Added the `array_filters` option.
        .. versionchanged:: 3.5
           Added the `collation` option.
        """
        super().__init__(filter, update, upsert, collation, array_filters, hint)

    def _add_to_bulk(self, bulkobj: _Bulk) -> None:
        """Add this operation to the _Bulk instance `bulkobj`."""
        bulkobj.add_update(self._filter, self._doc, False, self._upsert, collation=validate_collation_or_none(self._collation), array_filters=self._array_filters, hint=self._hint)