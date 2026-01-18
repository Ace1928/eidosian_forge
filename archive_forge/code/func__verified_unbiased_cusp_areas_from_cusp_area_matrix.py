from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _verified_unbiased_cusp_areas_from_cusp_area_matrix(cusp_area_matrix):
    num_cusps = cusp_area_matrix.dimensions()[0]
    result = num_cusps * [None]
    while None in result:
        stoppers = _find_stoppers(cusp_area_matrix, result)
        stoppers_union = _union_intervals([stopper[0] for stopper in stoppers])
        cusps = _get_cusps_from_stoppers(stoppers, result)
        stopper_pairs = set([stopper[1] for stopper in stoppers])
        stop_size = stoppers_union * stoppers_union / stoppers_union
        for area in result:
            if not area < stop_size:
                raise Exception('New area smaller than existing areas')
        for cusp in cusps:
            result[cusp] = stop_size
        for i in range(num_cusps):
            for j in range(i, num_cusps):
                if i in cusps or j in cusps:
                    if not result[i] is None and (not result[j] is None):
                        if not (i, j) in stopper_pairs:
                            if not result[i] * result[j] < cusp_area_matrix[i, j]:
                                raise Exception('Violated maximal cusp area', i, j)
    return result