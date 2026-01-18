from math import copysign, cos, hypot, isclose, pi
from fontTools.misc.roundTools import otRound
def _rounding_offset(direction):
    uv = _unit_vector(direction)
    if not uv:
        return (0, 0)
    result = []
    for uv_component in uv:
        if -_UNIT_VECTOR_THRESHOLD <= uv_component < _UNIT_VECTOR_THRESHOLD:
            result.append(0)
        else:
            result.append(copysign(1.0, uv_component))
    return tuple(result)