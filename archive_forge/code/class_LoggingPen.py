from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class LoggingPen(LogMixin, AbstractPen):
    """A pen with a ``log`` property (see fontTools.misc.loggingTools.LogMixin)"""
    pass