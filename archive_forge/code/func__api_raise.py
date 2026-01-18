from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
from oslo_db import api
from oslo_db import exception
from oslo_db.tests import base as test_base
def _api_raise(self, *args, **kwargs):
    """Simulate raising a database-has-gone-away error

        This method creates a fake OperationalError with an ID matching
        a valid MySQL "database has gone away" situation. It also decrements
        the error_counter so that we can artificially keep track of
        how many times this function is called by the wrapper. When
        error_counter reaches zero, this function returns True, simulating
        the database becoming available again and the query succeeding.
        """
    if self.error_counter > 0:
        self.error_counter -= 1
        orig = sqla.exc.DBAPIError(False, False, False)
        orig.args = [2006, 'Test raise operational error']
        e = exception.DBConnectionError(orig)
        raise e
    else:
        return True