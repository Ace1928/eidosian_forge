from enum import Enum, IntFlag
class GeodIntermediateFlag(IntFlag):
    """
    .. versionadded:: 3.1.0

    Flags to be used in Geod.[inv|fwd]_intermediate()
    """
    DEFAULT = 0
    NPTS_ROUND = 0
    NPTS_CEIL = 1
    NPTS_TRUNC = 2
    DEL_S_RECALC = 0
    DEL_S_NO_RECALC = 16
    AZIS_DISCARD = 0
    AZIS_KEEP = 256