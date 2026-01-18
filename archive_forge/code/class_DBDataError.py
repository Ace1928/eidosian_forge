from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class DBDataError(DBError):
    """Raised for errors that are due to problems with the processed data.

    E.g. division by zero, numeric value out of range, incorrect data type, etc

    """