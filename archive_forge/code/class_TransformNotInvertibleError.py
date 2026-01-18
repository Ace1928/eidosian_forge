from collections import namedtuple
import math
import warnings
class TransformNotInvertibleError(AffineError):
    """The transform could not be inverted"""