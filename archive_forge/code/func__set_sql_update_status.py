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
def _set_sql_update_status(self, enabled: Optional[bool]=True):
    """
        Sets the sql update status
        """
    if enabled:
        if not self.__pydantic_private__:
            self.__pydantic_private__ = {}
        self.__pydantic_private__['__in_sqlupdate__'] = True
    elif self.__pydantic_private__:
        _ = self.__pydantic_private__.pop('__in_sqlupdate__', None)