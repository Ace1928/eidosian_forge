import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('*', UnicodeEncodeError, '.*')
def _raise_for_unicode_encode(error, match, engine_name, is_disconnect):
    raise exception.DBInvalidUnicodeParameter()