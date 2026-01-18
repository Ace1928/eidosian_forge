from __future__ import annotations
import collections
import functools
import operator
import typing
from typing import Any
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .result import IteratorResult
from .result import MergedResult
from .result import Result
from .result import ResultMetaData
from .result import SimpleResultMetaData
from .result import tuplegetter
from .row import Row
from .. import exc
from .. import util
from ..sql import elements
from ..sql import sqltypes
from ..sql import util as sql_util
from ..sql.base import _generative
from ..sql.compiler import ResultColumnsEntry
from ..sql.compiler import RM_NAME
from ..sql.compiler import RM_OBJECTS
from ..sql.compiler import RM_RENDERED_NAME
from ..sql.compiler import RM_TYPE
from ..sql.type_api import TypeEngine
from ..util import compat
from ..util.typing import Literal
from ..util.typing import Self
class CursorResultMetaData(ResultMetaData):
    """Result metadata for DBAPI cursors."""
    __slots__ = ('_keymap', '_processors', '_keys', '_keymap_by_result_column_idx', '_tuplefilter', '_translated_indexes', '_safe_for_cache', '_unpickled', '_key_to_index')
    _keymap: _CursorKeyMapType
    _processors: _ProcessorsType
    _keymap_by_result_column_idx: Optional[Dict[int, _KeyMapRecType]]
    _unpickled: bool
    _safe_for_cache: bool
    _translated_indexes: Optional[List[int]]
    returns_rows: ClassVar[bool] = True

    def _has_key(self, key: Any) -> bool:
        return key in self._keymap

    def _for_freeze(self) -> ResultMetaData:
        return SimpleResultMetaData(self._keys, extra=[self._keymap[key][MD_OBJECTS] for key in self._keys])

    def _make_new_metadata(self, *, unpickled: bool, processors: _ProcessorsType, keys: Sequence[str], keymap: _KeyMapType, tuplefilter: Optional[_TupleGetterType], translated_indexes: Optional[List[int]], safe_for_cache: bool, keymap_by_result_column_idx: Any) -> CursorResultMetaData:
        new_obj = self.__class__.__new__(self.__class__)
        new_obj._unpickled = unpickled
        new_obj._processors = processors
        new_obj._keys = keys
        new_obj._keymap = keymap
        new_obj._tuplefilter = tuplefilter
        new_obj._translated_indexes = translated_indexes
        new_obj._safe_for_cache = safe_for_cache
        new_obj._keymap_by_result_column_idx = keymap_by_result_column_idx
        new_obj._key_to_index = self._make_key_to_index(keymap, MD_INDEX)
        return new_obj

    def _remove_processors(self) -> CursorResultMetaData:
        assert not self._tuplefilter
        return self._make_new_metadata(unpickled=self._unpickled, processors=[None] * len(self._processors), tuplefilter=None, translated_indexes=None, keymap={key: value[0:5] + (None,) + value[6:] for key, value in self._keymap.items()}, keys=self._keys, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx=self._keymap_by_result_column_idx)

    def _splice_horizontally(self, other: CursorResultMetaData) -> CursorResultMetaData:
        assert not self._tuplefilter
        keymap = dict(self._keymap)
        offset = len(self._keys)
        keymap.update({key: (value[0] + offset if value[0] is not None and key not in keymap else None, value[1] + offset, *value[2:]) for key, value in other._keymap.items()})
        return self._make_new_metadata(unpickled=self._unpickled, processors=self._processors + other._processors, tuplefilter=None, translated_indexes=None, keys=self._keys + other._keys, keymap=keymap, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx={metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry for metadata_entry in keymap.values()})

    def _reduce(self, keys: Sequence[_KeyIndexType]) -> ResultMetaData:
        recs = list(self._metadata_for_keys(keys))
        indexes = [rec[MD_INDEX] for rec in recs]
        new_keys: List[str] = [rec[MD_LOOKUP_KEY] for rec in recs]
        if self._translated_indexes:
            indexes = [self._translated_indexes[idx] for idx in indexes]
        tup = tuplegetter(*indexes)
        new_recs = [(index,) + rec[1:] for index, rec in enumerate(recs)]
        keymap = {rec[MD_LOOKUP_KEY]: rec for rec in new_recs}
        keymap.update(((e, new_rec) for new_rec in new_recs for e in new_rec[MD_OBJECTS] or ()))
        return self._make_new_metadata(unpickled=self._unpickled, processors=self._processors, keys=new_keys, tuplefilter=tup, translated_indexes=indexes, keymap=keymap, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx=self._keymap_by_result_column_idx)

    def _adapt_to_context(self, context: ExecutionContext) -> ResultMetaData:
        """When using a cached Compiled construct that has a _result_map,
        for a new statement that used the cached Compiled, we need to ensure
        the keymap has the Column objects from our new statement as keys.
        So here we rewrite keymap with new entries for the new columns
        as matched to those of the cached statement.

        """
        if not context.compiled or not context.compiled._result_columns:
            return self
        compiled_statement = context.compiled.statement
        invoked_statement = context.invoked_statement
        if TYPE_CHECKING:
            assert isinstance(invoked_statement, elements.ClauseElement)
        if compiled_statement is invoked_statement:
            return self
        assert invoked_statement is not None
        keymap_by_position = self._keymap_by_result_column_idx
        if keymap_by_position is None:
            keymap_by_position = self._keymap_by_result_column_idx = {metadata_entry[MD_RESULT_MAP_INDEX]: metadata_entry for metadata_entry in self._keymap.values()}
        assert not self._tuplefilter
        return self._make_new_metadata(keymap=compat.dict_union(self._keymap, {new: keymap_by_position[idx] for idx, new in enumerate(invoked_statement._all_selected_columns) if idx in keymap_by_position}), unpickled=self._unpickled, processors=self._processors, tuplefilter=None, translated_indexes=None, keys=self._keys, safe_for_cache=self._safe_for_cache, keymap_by_result_column_idx=self._keymap_by_result_column_idx)

    def __init__(self, parent: CursorResult[Any], cursor_description: _DBAPICursorDescription):
        context = parent.context
        self._tuplefilter = None
        self._translated_indexes = None
        self._safe_for_cache = self._unpickled = False
        if context.result_column_struct:
            result_columns, cols_are_ordered, textual_ordered, ad_hoc_textual, loose_column_name_matching = context.result_column_struct
            num_ctx_cols = len(result_columns)
        else:
            result_columns = cols_are_ordered = num_ctx_cols = ad_hoc_textual = loose_column_name_matching = textual_ordered = False
        raw = self._merge_cursor_description(context, cursor_description, result_columns, num_ctx_cols, cols_are_ordered, textual_ordered, ad_hoc_textual, loose_column_name_matching)
        self._processors = [metadata_entry[MD_PROCESSOR] for metadata_entry in raw]
        self._keymap_by_result_column_idx = None
        if num_ctx_cols:
            by_key = {metadata_entry[MD_LOOKUP_KEY]: metadata_entry for metadata_entry in raw}
            if len(by_key) != num_ctx_cols:
                index_by_key: Dict[Any, Any] = {}
                dupes = set()
                for metadata_entry in raw:
                    for key in (metadata_entry[MD_RENDERED_NAME],) + (metadata_entry[MD_OBJECTS] or ()):
                        idx = metadata_entry[MD_INDEX]
                        if index_by_key.setdefault(key, idx) != idx:
                            dupes.add(key)
                self._keymap = {obj_elem: metadata_entry for metadata_entry in raw if metadata_entry[MD_OBJECTS] for obj_elem in metadata_entry[MD_OBJECTS] if obj_elem not in dupes}
                by_key.update({key: (None, None, [], key, key, None, None) for key in dupes})
            else:
                self._keymap = {obj_elem: metadata_entry for metadata_entry in raw if metadata_entry[MD_OBJECTS] for obj_elem in metadata_entry[MD_OBJECTS]}
            self._keymap.update(by_key)
        else:
            self._keymap = {metadata_entry[MD_LOOKUP_KEY]: metadata_entry for metadata_entry in raw}
        if not num_ctx_cols and context._translate_colname:
            self._keymap.update({metadata_entry[MD_UNTRANSLATED]: self._keymap[metadata_entry[MD_LOOKUP_KEY]] for metadata_entry in raw if metadata_entry[MD_UNTRANSLATED]})
        self._key_to_index = self._make_key_to_index(self._keymap, MD_INDEX)

    def _merge_cursor_description(self, context, cursor_description, result_columns, num_ctx_cols, cols_are_ordered, textual_ordered, ad_hoc_textual, loose_column_name_matching):
        """Merge a cursor.description with compiled result column information.

        There are at least four separate strategies used here, selected
        depending on the type of SQL construct used to start with.

        The most common case is that of the compiled SQL expression construct,
        which generated the column names present in the raw SQL string and
        which has the identical number of columns as were reported by
        cursor.description.  In this case, we assume a 1-1 positional mapping
        between the entries in cursor.description and the compiled object.
        This is also the most performant case as we disregard extracting /
        decoding the column names present in cursor.description since we
        already have the desired name we generated in the compiled SQL
        construct.

        The next common case is that of the completely raw string SQL,
        such as passed to connection.execute().  In this case we have no
        compiled construct to work with, so we extract and decode the
        names from cursor.description and index those as the primary
        result row target keys.

        The remaining fairly common case is that of the textual SQL
        that includes at least partial column information; this is when
        we use a :class:`_expression.TextualSelect` construct.
        This construct may have
        unordered or ordered column information.  In the ordered case, we
        merge the cursor.description and the compiled construct's information
        positionally, and warn if there are additional description names
        present, however we still decode the names in cursor.description
        as we don't have a guarantee that the names in the columns match
        on these.   In the unordered case, we match names in cursor.description
        to that of the compiled construct based on name matching.
        In both of these cases, the cursor.description names and the column
        expression objects and names are indexed as result row target keys.

        The final case is much less common, where we have a compiled
        non-textual SQL expression construct, but the number of columns
        in cursor.description doesn't match what's in the compiled
        construct.  We make the guess here that there might be textual
        column expressions in the compiled construct that themselves include
        a comma in them causing them to split.  We do the same name-matching
        as with textual non-ordered columns.

        The name-matched system of merging is the same as that used by
        SQLAlchemy for all cases up through the 0.9 series.   Positional
        matching for compiled SQL expressions was introduced in 1.0 as a
        major performance feature, and positional matching for textual
        :class:`_expression.TextualSelect` objects in 1.1.
        As name matching is no longer
        a common case, it was acceptable to factor it into smaller generator-
        oriented methods that are easier to understand, but incur slightly
        more performance overhead.

        """
        if num_ctx_cols and cols_are_ordered and (not textual_ordered) and (num_ctx_cols == len(cursor_description)):
            self._keys = [elem[0] for elem in result_columns]
            self._safe_for_cache = True
            return [(idx, idx, rmap_entry[RM_OBJECTS], rmap_entry[RM_NAME], rmap_entry[RM_RENDERED_NAME], context.get_result_processor(rmap_entry[RM_TYPE], rmap_entry[RM_RENDERED_NAME], cursor_description[idx][1]), None) for idx, rmap_entry in enumerate(result_columns)]
        else:
            if textual_ordered or (ad_hoc_textual and len(cursor_description) == num_ctx_cols):
                self._safe_for_cache = True
                raw_iterator = self._merge_textual_cols_by_position(context, cursor_description, result_columns)
            elif num_ctx_cols:
                self._safe_for_cache = False
                raw_iterator = self._merge_cols_by_name(context, cursor_description, result_columns, loose_column_name_matching)
            else:
                self._safe_for_cache = False
                raw_iterator = self._merge_cols_by_none(context, cursor_description)
            return [(idx, ridx, obj, cursor_colname, cursor_colname, context.get_result_processor(mapped_type, cursor_colname, coltype), untranslated) for idx, ridx, cursor_colname, mapped_type, coltype, obj, untranslated in raw_iterator]

    def _colnames_from_description(self, context, cursor_description):
        """Extract column names and data types from a cursor.description.

        Applies unicode decoding, column translation, "normalization",
        and case sensitivity rules to the names based on the dialect.

        """
        dialect = context.dialect
        translate_colname = context._translate_colname
        normalize_name = dialect.normalize_name if dialect.requires_name_normalize else None
        untranslated = None
        self._keys = []
        for idx, rec in enumerate(cursor_description):
            colname = rec[0]
            coltype = rec[1]
            if translate_colname:
                colname, untranslated = translate_colname(colname)
            if normalize_name:
                colname = normalize_name(colname)
            self._keys.append(colname)
            yield (idx, colname, untranslated, coltype)

    def _merge_textual_cols_by_position(self, context, cursor_description, result_columns):
        num_ctx_cols = len(result_columns)
        if num_ctx_cols > len(cursor_description):
            util.warn('Number of columns in textual SQL (%d) is smaller than number of columns requested (%d)' % (num_ctx_cols, len(cursor_description)))
        seen = set()
        for idx, colname, untranslated, coltype in self._colnames_from_description(context, cursor_description):
            if idx < num_ctx_cols:
                ctx_rec = result_columns[idx]
                obj = ctx_rec[RM_OBJECTS]
                ridx = idx
                mapped_type = ctx_rec[RM_TYPE]
                if obj[0] in seen:
                    raise exc.InvalidRequestError('Duplicate column expression requested in textual SQL: %r' % obj[0])
                seen.add(obj[0])
            else:
                mapped_type = sqltypes.NULLTYPE
                obj = None
                ridx = None
            yield (idx, ridx, colname, mapped_type, coltype, obj, untranslated)

    def _merge_cols_by_name(self, context, cursor_description, result_columns, loose_column_name_matching):
        match_map = self._create_description_match_map(result_columns, loose_column_name_matching)
        mapped_type: TypeEngine[Any]
        for idx, colname, untranslated, coltype in self._colnames_from_description(context, cursor_description):
            try:
                ctx_rec = match_map[colname]
            except KeyError:
                mapped_type = sqltypes.NULLTYPE
                obj = None
                result_columns_idx = None
            else:
                obj = ctx_rec[1]
                mapped_type = ctx_rec[2]
                result_columns_idx = ctx_rec[3]
            yield (idx, result_columns_idx, colname, mapped_type, coltype, obj, untranslated)

    @classmethod
    def _create_description_match_map(cls, result_columns: List[ResultColumnsEntry], loose_column_name_matching: bool=False) -> Dict[Union[str, object], Tuple[str, Tuple[Any, ...], TypeEngine[Any], int]]:
        """when matching cursor.description to a set of names that are present
        in a Compiled object, as is the case with TextualSelect, get all the
        names we expect might match those in cursor.description.
        """
        d: Dict[Union[str, object], Tuple[str, Tuple[Any, ...], TypeEngine[Any], int]] = {}
        for ridx, elem in enumerate(result_columns):
            key = elem[RM_RENDERED_NAME]
            if key in d:
                e_name, e_obj, e_type, e_ridx = d[key]
                d[key] = (e_name, e_obj + elem[RM_OBJECTS], e_type, ridx)
            else:
                d[key] = (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx)
            if loose_column_name_matching:
                for r_key in elem[RM_OBJECTS]:
                    d.setdefault(r_key, (elem[RM_NAME], elem[RM_OBJECTS], elem[RM_TYPE], ridx))
        return d

    def _merge_cols_by_none(self, context, cursor_description):
        for idx, colname, untranslated, coltype in self._colnames_from_description(context, cursor_description):
            yield (idx, None, colname, sqltypes.NULLTYPE, coltype, None, untranslated)
    if not TYPE_CHECKING:

        def _key_fallback(self, key: Any, err: Optional[Exception], raiseerr: bool=True) -> Optional[NoReturn]:
            if raiseerr:
                if self._unpickled and isinstance(key, elements.ColumnElement):
                    raise exc.NoSuchColumnError('Row was unpickled; lookup by ColumnElement is unsupported') from err
                else:
                    raise exc.NoSuchColumnError("Could not locate column in row for column '%s'" % util.string_or_unprintable(key)) from err
            else:
                return None

    def _raise_for_ambiguous_column_name(self, rec):
        raise exc.InvalidRequestError("Ambiguous column name '%s' in result set column descriptions" % rec[MD_LOOKUP_KEY])

    def _index_for_key(self, key: Any, raiseerr: bool=True) -> Optional[int]:
        if isinstance(key, int):
            key = self._keys[key]
        try:
            rec = self._keymap[key]
        except KeyError as ke:
            x = self._key_fallback(key, ke, raiseerr)
            assert x is None
            return None
        index = rec[0]
        if index is None:
            self._raise_for_ambiguous_column_name(rec)
        return index

    def _indexes_for_keys(self, keys):
        try:
            return [self._keymap[key][0] for key in keys]
        except KeyError as ke:
            CursorResultMetaData._key_fallback(self, ke.args[0], ke)

    def _metadata_for_keys(self, keys: Sequence[Any]) -> Iterator[_NonAmbigCursorKeyMapRecType]:
        for key in keys:
            if int in key.__class__.__mro__:
                key = self._keys[key]
            try:
                rec = self._keymap[key]
            except KeyError as ke:
                CursorResultMetaData._key_fallback(self, ke.args[0], ke)
            index = rec[MD_INDEX]
            if index is None:
                self._raise_for_ambiguous_column_name(rec)
            yield cast(_NonAmbigCursorKeyMapRecType, rec)

    def __getstate__(self):
        return {'_keymap': {key: (rec[MD_INDEX], rec[MD_RESULT_MAP_INDEX], [], key, rec[MD_RENDERED_NAME], None, None) for key, rec in self._keymap.items() if isinstance(key, (str, int))}, '_keys': self._keys, '_translated_indexes': self._translated_indexes}

    def __setstate__(self, state):
        self._processors = [None for _ in range(len(state['_keys']))]
        self._keymap = state['_keymap']
        self._keymap_by_result_column_idx = None
        self._key_to_index = self._make_key_to_index(self._keymap, MD_INDEX)
        self._keys = state['_keys']
        self._unpickled = True
        if state['_translated_indexes']:
            self._translated_indexes = cast('List[int]', state['_translated_indexes'])
            self._tuplefilter = tuplegetter(*self._translated_indexes)
        else:
            self._translated_indexes = self._tuplefilter = None