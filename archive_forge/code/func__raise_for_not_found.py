from __future__ import annotations
import datetime
from pydantic import BaseModel
from pydantic.alias_generators import to_snake
from dataclasses import dataclass
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import Mapped
from sqlalchemy import MetaData
from sqlalchemy.orm import mapped_column
from sqlalchemy import Text, Table
from sqlalchemy import func as sql_func
from sqlalchemy.orm import defer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text, select, Select, ColumnElement, and_, update, Update, delete, or_
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import registry
from ...types import errors
from typing import Optional, Type, TypeVar, Union, Set, Any, Tuple, List, Dict, cast, Generic, Generator, Callable, TYPE_CHECKING
def _raise_for_not_found(self, obj: Optional[ModelType], raise_for_not_found: bool) -> None:
    """
        Raises an error if the object is not found
        """
    if not obj and raise_for_not_found:
        raise errors.DatabaseItemNotFoundException(self.object_class_name)