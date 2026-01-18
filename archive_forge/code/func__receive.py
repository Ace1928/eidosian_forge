import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
def _receive(fn):
    _registry[dbname][exception_type].extend(((fn, re.compile(reg, re.DOTALL)) for reg in ((regex,) if not isinstance(regex, tuple) else regex)))
    return fn