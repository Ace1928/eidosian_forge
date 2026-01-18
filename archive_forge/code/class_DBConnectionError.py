from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBConnectionError(DBError):
    """Wrapped connection specific exception.

    Raised when database connection is failed.
    """
    pass