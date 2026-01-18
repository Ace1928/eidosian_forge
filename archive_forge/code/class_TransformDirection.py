from enum import Enum, IntFlag
class TransformDirection(BaseEnum):
    """
    .. versionadded:: 2.2.0

    Supported transform directions
    """
    FORWARD = 'FORWARD'
    INVERSE = 'INVERSE'
    IDENT = 'IDENT'