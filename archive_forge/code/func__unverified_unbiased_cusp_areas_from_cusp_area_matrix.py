from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _unverified_unbiased_cusp_areas_from_cusp_area_matrix(cusp_area_matrix):
    num_cusps = cusp_area_matrix.dimensions()[0]
    num_pending = num_cusps
    result = num_cusps * [None]
    while num_pending > 0:
        stop_size, (i, j) = _find_minimal_stopper(cusp_area_matrix, result)
        if result[i] is None:
            result[i] = stop_size
            num_pending -= 1
        if i != j:
            if result[j] is None:
                result[j] = stop_size
                num_pending -= 1
    return result