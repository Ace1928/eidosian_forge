from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement
def _find_stoppers(cusp_area_matrix, assigned_areas):
    return _interval_minimum_candidates(_find_potential_stoppers(cusp_area_matrix, assigned_areas))