import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _transform_barycentric_cartesian():
    """
    Returns the transformation matrix from barycentric to Cartesian
    coordinates and conversely.
    """
    tri_verts = np.array([[0.5, np.sqrt(3) / 2], [0, 0], [1, 0]])
    M = np.array([tri_verts[:, 0], tri_verts[:, 1], np.ones(3)])
    return (M, np.linalg.inv(M))