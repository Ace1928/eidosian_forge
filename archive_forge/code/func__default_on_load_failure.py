from .exception import MultipleMatches
from .exception import NoMatches
from .named import NamedExtensionManager
@staticmethod
def _default_on_load_failure(drivermanager, ep, err):
    raise