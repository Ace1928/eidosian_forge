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
@classmethod
def _teardown_once_metadata_bind(cls):
    if cls.run_create_tables:
        drop_all_tables_from_metadata(cls._tables_metadata, cls.bind)
    if cls.run_dispose_bind == 'once':
        cls.dispose_bind(cls.bind)
    cls._tables_metadata.bind = None
    if cls.run_setup_bind is not None:
        cls.bind = None