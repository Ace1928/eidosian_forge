import contextlib
from sqlalchemy import func, text
from sqlalchemy import delete as sqlalchemy_delete
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy import exists as sqlalchemy_exists
from sqlalchemy.future import select
from sqlalchemy.sql.expression import Select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload, joinedload, immediateload
from sqlalchemy import Column, Integer, DateTime, String, Text, ForeignKey, Boolean, Identity, Enum
from typing import Any, Generator, AsyncGenerator, Iterable, Optional, Union, Type, Dict, cast, TYPE_CHECKING, List, Tuple, TypeVar, Callable
from lazyops.utils import create_unique_id, create_timestamp
from lazyops.utils.logs import logger
from lazyops.types import lazyproperty
from lazyops.libs.psqldb.base import Base, PostgresDB, AsyncSession, Session
from lazyops.libs.psqldb.utils import SQLJson, get_pydantic_model, object_serializer, get_sqlmodel_dict
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
@classmethod
def _nested_kwarg_parser(cls: SQLModelT, top_level_obj: Optional[SQLModelT]=None, **kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        Parses nested dict kwargs into a flat dict
        """
    attrs = []
    for key, value in kwargs.items():
        obj_attr = getattr(cls, key, None) if top_level_obj is None else getattr(top_level_obj, key, None)
        if obj_attr is None:
            continue
        if isinstance(value, dict):
            attrs.append(obj_attr.has(**value))
        else:
            attrs.append(obj_attr == value)
    logger.warning(f'{attrs}')
    return attrs