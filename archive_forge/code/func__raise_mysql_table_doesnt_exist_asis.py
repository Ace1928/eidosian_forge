import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.DBAPIError, '.*\\b1146\\b')
def _raise_mysql_table_doesnt_exist_asis(error, match, engine_name, is_disconnect):
    """Raise MySQL error 1146 as is.

    Raise MySQL error 1146 as is, so that it does not conflict with
    the MySQL dialect's checking a table not existing.
    """
    raise error