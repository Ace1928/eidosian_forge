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
def _supdate(self, session: Optional[Session]=None, **kwargs):
    """
        Update a record
        """
    with PostgresDB.session(session=session) as db_sess:
        query = sqlalchemy_update(self.__class__).where(self.__class__.id == self.id).values(**kwargs)
        db_sess.execute(query)
        db_sess.commit()
    return self._get(id=self.id)