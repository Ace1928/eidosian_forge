from oslo_utils.excutils import CausedByException
from oslo_db._i18n import _
class CantStartEngineError(Exception):
    """Error raised when the enginefacade cannot start up correctly."""