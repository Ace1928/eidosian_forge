from __future__ import annotations
from typing import (
from bson.raw_bson import RawBSONDocument
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import validate_boolean, validate_is_mapping, validate_list
from pymongo.helpers import _gen_index_name, _index_document, _index_list
from pymongo.typings import _CollationIn, _DocumentType, _Pipeline
class ReplaceOne(Generic[_DocumentType]):
    """Represents a replace_one operation."""
    __slots__ = ('_filter', '_doc', '_upsert', '_collation', '_hint')

    def __init__(self, filter: Mapping[str, Any], replacement: Union[_DocumentType, RawBSONDocument], upsert: bool=False, collation: Optional[_CollationIn]=None, hint: Optional[_IndexKeyHint]=None) -> None:
        """Create a ReplaceOne instance.

        For use with :meth:`~pymongo.collection.Collection.bulk_write`.

        :Parameters:
          - `filter`: A query that matches the document to replace.
          - `replacement`: The new document.
          - `upsert` (optional): If ``True``, perform an insert if no documents
            match the filter.
          - `collation` (optional): An instance of
            :class:`~pymongo.collation.Collation`.
          - `hint` (optional): An index to use to support the query
            predicate specified either by its string name, or in the same
            format as passed to
            :meth:`~pymongo.collection.Collection.create_index` (e.g.
            ``[('field', ASCENDING)]``). This option is only supported on
            MongoDB 4.2 and above.

        .. versionchanged:: 3.11
           Added the ``hint`` option.
        .. versionchanged:: 3.5
           Added the ``collation`` option.
        """
        if filter is not None:
            validate_is_mapping('filter', filter)
        if upsert is not None:
            validate_boolean('upsert', upsert)
        if hint is not None and (not isinstance(hint, str)):
            self._hint: Union[str, SON[str, Any], None] = helpers._index_document(hint)
        else:
            self._hint = hint
        self._filter = filter
        self._doc = replacement
        self._upsert = upsert
        self._collation = collation

    def _add_to_bulk(self, bulkobj: _Bulk) -> None:
        """Add this operation to the _Bulk instance `bulkobj`."""
        bulkobj.add_replace(self._filter, self._doc, self._upsert, collation=validate_collation_or_none(self._collation), hint=self._hint)

    def __eq__(self, other: Any) -> bool:
        if type(other) == type(self):
            return (other._filter, other._doc, other._upsert, other._collation, other._hint) == (self._filter, self._doc, self._upsert, self._collation, other._hint)
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return '{}({!r}, {!r}, {!r}, {!r}, {!r})'.format(self.__class__.__name__, self._filter, self._doc, self._upsert, self._collation, self._hint)