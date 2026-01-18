import datetime
import functools
import pytz
from oslo_db import exception as db_exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import models
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from osprofiler import opts as profiler
import osprofiler.sqlalchemy
import sqlalchemy as sql
from sqlalchemy.ext import declarative
from sqlalchemy.orm.attributes import flag_modified, InstrumentedAttribute
from sqlalchemy import types as sql_types
from keystone.common import driver_hints
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class _WontMatch(Exception):
    """Raised to indicate that the filter won't match.

    This is raised to short-circuit the computation of the filter as soon as
    it's discovered that the filter requested isn't going to match anything.

    A filter isn't going to match anything if the value is too long for the
    field, for example.

    """

    @classmethod
    def check(cls, value, col_attr):
        """Check if the value can match given the column attributes.

        Raises this class if the value provided can't match any value in the
        column in the table given the column's attributes. For example, if the
        column is a string and the value is longer than the column then it
        won't match any value in the column in the table.

        """
        if value is None:
            return
        col = col_attr.property.columns[0]
        if isinstance(col.type, sql.types.Boolean):
            return
        if not col.type.length:
            return
        if len(value) > col.type.length:
            raise cls()