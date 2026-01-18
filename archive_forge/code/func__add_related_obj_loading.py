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
def _add_related_obj_loading(self, stmt: Select, load_children: List[str]=None) -> Select:
    """
        Constructs a select statement with the given related objects
        """
    if load_children:
        options = iter((getattr(self.model, attr) for attr in load_children))
        stmt = stmt.options(selectinload(*options))
    return stmt