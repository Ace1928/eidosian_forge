from ...sage_helper import _within_sage, sage_method

    Given lifted Ptolemy coordinates for a triangulation (as dictionary)
    and the number of tetrahedra, compute the complex volume (where
    the real part is the Chern-Simons and the imaginary part is the
    volume).

    This sums of the dilogs across tetrahedra without adjusting for the
    fact that the triangulation might not be ordered.
    Thus, the Chern-Simons is correct only up to multiples of pi^2/6.
    