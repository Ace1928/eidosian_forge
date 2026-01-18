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
def _setup_each_mappers(self):
    if self.run_setup_mappers != 'once':
        self.__class__.mapper_registry, self.__class__.mapper = self._generate_registry()
    if self.run_setup_mappers == 'each':
        self._with_register_classes(self.setup_mappers)