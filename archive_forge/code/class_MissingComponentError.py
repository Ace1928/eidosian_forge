from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
class MissingComponentError(KeyError):
    """Indicates a component pointing to a non-existent glyph in the glyphset."""