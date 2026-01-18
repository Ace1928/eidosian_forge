from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _find_potential_stoppers(cusp_area_matrix, assigned_areas):

    def stopper(i, j):
        if not assigned_areas[i] is None:
            return cusp_area_matrix[i, j] / assigned_areas[i]
        if not assigned_areas[j] is None:
            return cusp_area_matrix[i, j] / assigned_areas[j]
        return sqrt(cusp_area_matrix[i, j])
    num_cusps = cusp_area_matrix.dimensions()[0]
    return [(stopper(i, j), (i, j)) for i in range(num_cusps) for j in range(i, num_cusps) if assigned_areas[j] is None or assigned_areas[i] is None]