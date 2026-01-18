import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _replace_zero_coords(ternary_data, delta=0.0005):
    """
    Replaces zero ternary coordinates with delta and normalize the new
    triplets (a, b, c).

    Parameters
    ----------

    ternary_data : ndarray of shape (N, 3)

    delta : float
        Small float to regularize logarithm.

    Notes
    -----
    Implements a method
    by J. A. Martin-Fernandez,  C. Barcelo-Vidal, V. Pawlowsky-Glahn,
    Dealing with zeros and missing values in compositional data sets
    using nonparametric imputation, Mathematical Geology 35 (2003),
    pp 253-278.
    """
    zero_mask = ternary_data == 0
    is_any_coord_zero = np.any(zero_mask, axis=0)
    unity_complement = 1 - delta * is_any_coord_zero
    if np.any(unity_complement) < 0:
        raise ValueError('The provided value of delta led to negativeternary coords.Set a smaller delta')
    ternary_data = np.where(zero_mask, delta, unity_complement * ternary_data)
    return ternary_data