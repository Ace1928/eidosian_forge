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
def _setup_each_inserts(self):
    if self.run_inserts == 'each':
        self._load_fixtures()
        with self.bind.begin() as conn:
            self.insert_data(conn)