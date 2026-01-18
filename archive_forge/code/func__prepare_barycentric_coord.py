import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _prepare_barycentric_coord(b_coords):
    """
    Check ternary coordinates and return the right barycentric coordinates.
    """
    if not isinstance(b_coords, (list, np.ndarray)):
        raise ValueError('Data  should be either an array of shape (n,m),or a list of n m-lists, m=2 or 3')
    b_coords = np.asarray(b_coords)
    if b_coords.shape[0] not in (2, 3):
        raise ValueError('A point should have  2 (a, b) or 3 (a, b, c)barycentric coordinates')
    if len(b_coords) == 3 and (not np.allclose(b_coords.sum(axis=0), 1, rtol=0.01)) and (not np.allclose(b_coords.sum(axis=0), 100, rtol=0.01)):
        msg = 'The sum of coordinates should be 1 or 100 for all data points'
        raise ValueError(msg)
    if len(b_coords) == 2:
        A, B = b_coords
        C = 1 - (A + B)
    else:
        A, B, C = b_coords / b_coords.sum(axis=0)
    if np.any(np.stack((A, B, C)) < 0):
        raise ValueError('Barycentric coordinates should be positive.')
    return np.stack((A, B, C))