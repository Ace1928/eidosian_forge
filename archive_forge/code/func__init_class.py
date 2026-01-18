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
def _init_class(cls):
    if cls.run_define_tables == 'each':
        if cls.run_create_tables == 'once':
            cls.run_create_tables = 'each'
        assert cls.run_inserts in ('each', None)
    cls.other = adict()
    cls.tables = adict()
    cls.sequences = adict()
    cls.bind = cls.setup_bind()
    cls._tables_metadata = sa.MetaData()