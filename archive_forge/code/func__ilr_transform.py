import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _ilr_transform(barycentric):
    """
    Perform Isometric Log-Ratio on barycentric (compositional) data.

    Parameters
    ----------
    barycentric: ndarray of shape (3, N)
        Barycentric coordinates.

    References
    ----------
    "An algebraic method to compute isometric logratio transformation and
    back transformation of compositional data", Jarauta-Bragulat, E.,
    Buenestado, P.; Hervada-Sala, C., in Proc. of the Annual Conf. of the
    Intl Assoc for Math Geology, 2003, pp 31-30.
    """
    barycentric = np.asarray(barycentric)
    x_0 = np.log(barycentric[0] / barycentric[1]) / np.sqrt(2)
    x_1 = 1.0 / np.sqrt(6) * np.log(barycentric[0] * barycentric[1] / barycentric[2] ** 2)
    ilr_tdata = np.stack((x_0, x_1))
    return ilr_tdata