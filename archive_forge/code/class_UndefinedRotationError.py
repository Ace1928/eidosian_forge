from collections import namedtuple
import math
import warnings
class UndefinedRotationError(AffineError):
    """The rotation angle could not be computed for this transform"""