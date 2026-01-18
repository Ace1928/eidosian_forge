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
@contextlib.contextmanager
def _sql_update_context(self):
    """
        Context Manager for the sql update
        """
    self._set_sql_update_status(enabled=True)
    try:
        yield
    finally:
        self._set_sql_update_status(enabled=False)