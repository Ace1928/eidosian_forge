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
def _get_export_sql_insert_fields(self, schemas: Dict[str, Union[str, List[str], Dict[str, str]]], data: Optional[Dict[str, Any]]=None, include: 'IncEx'=None, exclude: 'IncEx'=None, by_alias: bool=False, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, **kwargs) -> List[str]:
    """
        Gets the export sql index
        """
    if data is None:
        data = self.model_dump(mode='json', include=include, exclude=exclude, by_alias=by_alias, exclude_unset=exclude_unset, exclude_defaults=exclude_defaults, exclude_none=exclude_none)
    return [k for k in schemas['sql_keys'] if k in data]