from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class PenError(Exception):
    """Represents an error during penning."""