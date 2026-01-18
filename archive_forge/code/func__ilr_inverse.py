import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _ilr_inverse(x):
    """
    Perform inverse Isometric Log-Ratio (ILR) transform to retrieve
    barycentric (compositional) data.

    Parameters
    ----------
    x : array of shape (2, N)
        Coordinates in ILR space.

    References
    ----------
    "An algebraic method to compute isometric logratio transformation and
    back transformation of compositional data", Jarauta-Bragulat, E.,
    Buenestado, P.; Hervada-Sala, C., in Proc. of the Annual Conf. of the
    Intl Assoc for Math Geology, 2003, pp 31-30.
    """
    x = np.array(x)
    matrix = np.array([[0.5, 1, 1.0], [-0.5, 1, 1.0], [0.0, 0.0, 1.0]])
    s = np.sqrt(2) / 2
    t = np.sqrt(3 / 2)
    Sk = np.einsum('ik, kj -> ij', np.array([[s, t], [-s, t]]), x)
    Z = -np.log(1 + np.exp(Sk).sum(axis=0))
    log_barycentric = np.einsum('ik, kj -> ij', matrix, np.stack((2 * s * x[0], t * x[1], Z)))
    iilr_tdata = np.exp(log_barycentric)
    return iilr_tdata