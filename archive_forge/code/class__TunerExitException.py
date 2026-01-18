import re
from lightning_fabric.utilities.exceptions import MisconfigurationException  # noqa: F401
class _TunerExitException(Exception):
    """Exception used to exit early while tuning."""