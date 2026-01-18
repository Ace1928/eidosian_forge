import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('*', sqla_exc.OperationalError, '.*')
def _raise_operational_errors_directly_filter(operational_error, match, engine_name, is_disconnect):
    """Filter for all remaining OperationalError classes and apply.

    Filter for all remaining OperationalError classes and apply
    special rules.
    """
    if is_disconnect:
        raise exception.DBConnectionError(operational_error)
    else:
        raise operational_error