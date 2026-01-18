from ..sage_helper import _within_sage
from ..math_basics import correct_min, is_RealIntervalFieldElement


        sage: from sage.all import matrix, RIF
        sage: greedy_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[RIF(9.0,9.0005),RIF(6.0, 6.001)],
        ...                     [RIF(6.0,6.001 ),RIF(10.0, 10.001)]]))
        [3.0001?, 2.000?]

        >>> from snappy.SnapPy import matrix
        >>> greedy_cusp_areas_from_cusp_area_matrix(
        ...             matrix([[10.0, 40.0],
        ...                     [40.0, 20.0]]))
        [3.1622776601683795, 4.47213595499958]

    