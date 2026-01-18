import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('sqlite', sqla_exc.OperationalError, '.* no such table: (?P<table>.+)')
@filters('mysql', sqla_exc.InternalError, '.*1051,.*Unknown table \'(.+\\.)?(?P<table>.+)\'\\"')
@filters('mysql', sqla_exc.OperationalError, '.*1051,.*Unknown table \'(.+\\.)?(?P<table>.+)\'\\"')
@filters('postgresql', sqla_exc.ProgrammingError, '.* table \\"(?P<table>.+)\\" does not exist')
def _check_table_non_existing(programming_error, match, engine_name, is_disconnect):
    """Filter for table non existing errors."""
    raise exception.DBNonExistentTable(match.group('table'), programming_error)