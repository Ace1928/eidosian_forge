from __future__ import annotations
import weakref
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.asyncio import AsyncSession, async_object_session
class CarefulSession(orm.Session):

    def add(self, object_):
        object_session = orm.object_session(object_)
        if object_session and object_session is not self:
            object_session.expunge(object_)
            object_._prev_session = weakref.ref(object_session)
        return super().add(object_)