from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class ColumnError(Exception):
    """Error raised when no column or an invalid column is found."""