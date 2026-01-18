from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class RetryRequest(Exception):
    """Error raised when DB operation needs to be retried.

    That could be intentionally raised by the code without any real DB errors.
    """

    def __init__(self, inner_exc):
        self.inner_exc = inner_exc