from __future__ import annotations
import itertools
import random
import re
import sys
import sqlalchemy as sa
from .base import TestBase
from .. import config
from .. import mock
from ..assertions import eq_
from ..assertions import ne_
from ..util import adict
from ..util import drop_all_tables_from_metadata
from ... import event
from ... import util
from ...schema import sort_tables_and_constraints
from ...sql import visitors
from ...sql.elements import ClauseElement
@config.fixture(autouse=True, scope='class')
def _setup_tables_test_class(self):
    cls = self.__class__
    cls._init_class()
    cls._setup_once_tables()
    cls._setup_once_inserts()
    yield
    cls._teardown_once_metadata_bind()