from __future__ import annotations
from typing import Any
import sqlalchemy as sa
from .base import TestBase
from .sql import TablesTest
from .. import assertions
from .. import config
from .. import schema
from ..entities import BasicEntity
from ..entities import ComparableEntity
from ..util import adict
from ... import orm
from ...orm import DeclarativeBase
from ...orm import events as orm_events
from ...orm import registry
@config.fixture(autouse=True)
def _remove_listeners(self):
    yield
    orm_events.MapperEvents._clear()
    orm_events.InstanceEvents._clear()
    orm_events.SessionEvents._clear()
    orm_events.InstrumentationEvents._clear()
    orm_events.QueryEvents._clear()