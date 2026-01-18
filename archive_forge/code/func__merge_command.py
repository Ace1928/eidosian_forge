from __future__ import annotations
import copy
from collections.abc import MutableMapping
from itertools import islice
from typing import (
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo import _csot, common
from pymongo.client_session import ClientSession, _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES, _get_wce_doc
from pymongo.message import (
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def _merge_command(run: _Run, full_result: MutableMapping[str, Any], offset: int, result: Mapping[str, Any]) -> None:
    """Merge a write command result into the full bulk result."""
    affected = result.get('n', 0)
    if run.op_type == _INSERT:
        full_result['nInserted'] += affected
    elif run.op_type == _DELETE:
        full_result['nRemoved'] += affected
    elif run.op_type == _UPDATE:
        upserted = result.get('upserted')
        if upserted:
            n_upserted = len(upserted)
            for doc in upserted:
                doc['index'] = run.index(doc['index'] + offset)
            full_result['upserted'].extend(upserted)
            full_result['nUpserted'] += n_upserted
            full_result['nMatched'] += affected - n_upserted
        else:
            full_result['nMatched'] += affected
        full_result['nModified'] += result['nModified']
    write_errors = result.get('writeErrors')
    if write_errors:
        for doc in write_errors:
            replacement = doc.copy()
            idx = doc['index'] + offset
            replacement['index'] = run.index(idx)
            replacement['op'] = run.ops[idx]
            full_result['writeErrors'].append(replacement)
    wce = _get_wce_doc(result)
    if wce:
        full_result['writeConcernErrors'].append(wce)