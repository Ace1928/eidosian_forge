from __future__ import annotations
from lazyops.imports._sqlalchemy import require_sql
import datetime
from uuid import UUID
from pydantic import BaseModel
from pydantic.alias_generators import to_snake
from dataclasses import dataclass
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import as_declarative, declared_attr
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import MappedAsDataclass
from sqlalchemy.orm import Mapped, InstrumentedAttribute
from sqlalchemy.orm import mapped_column
from sqlalchemy import func as sql_func
from sqlalchemy import Text
from sqlalchemy.orm import defer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.inspection import inspect
from sqlalchemy.sql.expression import text, select, Select, ColumnElement, and_, update, Update, delete, or_
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.orm import selectinload
from sqlalchemy.orm import registry
from lazyops.utils.lazy import lazy_import
from lazyops.utils.logs import logger
from typing import Optional, Type, TypeVar, Union, Set, Any, Tuple, Literal, List, Dict, cast, Generic, Generator, Callable, TYPE_CHECKING
from . import errors
def _convert_source_to_orm_model(self, source: SourceSchemaType, model_dump_kwargs: Optional[Dict[str, Any]]=None, **kwargs) -> ModelTypeORM:
    """
        Converts the source to the ORM model
        """
    data = self._convert_source_to_dict(source, model_dump_kwargs=model_dump_kwargs, **kwargs)
    try:
        return self.model(**data)
    except Exception as e:
        logger.trace(f'Error converting {type(source)} -> {self.object_class_name}', e)
        raise e