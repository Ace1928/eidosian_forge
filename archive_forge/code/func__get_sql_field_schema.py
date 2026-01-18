from __future__ import annotations
import abc
import datetime
import contextlib
from pydantic import BaseModel, model_validator, Field, PrivateAttr, validator
from lazyops.utils.logs import logger
from .static import SqliteTemplates
from .registry import get_or_register_sqlite_schema, get_or_register_sqlite_connection, retrieve_sqlite_model_schema, get_sqlite_model_pkey, get_or_register_sqlite_tablename, SQLiteModelConfig, get_sqlite_model_config
from .utils import normalize_sql_text
from typing import Optional, List, Tuple, Dict, Union, TypeVar, Any, overload, TYPE_CHECKING
@classmethod
def _get_sql_field_schema(cls) -> Tuple[Dict[str, str], str, Dict[str, int]]:
    """
        Gets the sql field schema
        """
    results = {}
    if any((f.json_schema_extra and f.json_schema_extra.get('include_sql', False) for f in cls.model_fields.values())):
        field_keys = [name for name, f in cls.model_fields.items() if name != 'tablename' and (f.json_schema_extra and f.json_schema_extra.get('include_sql', False) and (not f.json_schema_extra.get('exclude_sql', False)))]
        logger.info(f'Model {cls.__name__} has `include_sql` fields. Only selecting fields: {field_keys}', prefix=f'|g|DB: {cls.__name__}|e|', colored=True)
    else:
        field_keys = [name for name, f in cls.model_fields.items() if name != 'tablename' and (not f.exclude) and (not f.json_schema_extra or not f.json_schema_extra.get('exclude_sql', False))]
    pkey = [name for name in field_keys if cls.model_fields[name].json_schema_extra and cls.model_fields[name].json_schema_extra.get('sql_pkey', False)]
    pkey = pkey[0] if pkey else field_keys[0]
    for name, field in cls.model_fields.items():
        if name == 'tablename':
            continue
        if name not in field_keys:
            continue
        if field.annotation in [str, Optional[str]]:
            results[name] = 'TEXT'
        elif field.annotation in [int, Optional[int]]:
            results[name] = 'INTEGER'
        elif field.annotation in [float, Optional[float]]:
            results[name] = 'REAL'
        elif field.annotation in [bool, Optional[bool]]:
            results[name] = 'BOOLEAN'
        elif field.annotation in [datetime.datetime, Optional[datetime.datetime]]:
            results[name] = 'DATETIME'
        else:
            logger.warning(f'Unsupported Field Type: {field.annotation}', prefix='|g|DB|e|', colored=True)
            continue
        if name == pkey:
            results[name] += ' PRIMARY KEY'
    search_precisions: Dict[str, int] = {name: cls.model_fields[name].json_schema_extra.get('sql_search_precision', 0) if cls.model_fields[name].json_schema_extra else 0 for name in field_keys}
    return (results, pkey, search_precisions)