import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _colors(ncontours, colormap=None):
    """
    Return a list of ``ncontours`` colors from the ``colormap`` colorscale.
    """
    if colormap in clrs.PLOTLY_SCALES.keys():
        cmap = clrs.PLOTLY_SCALES[colormap]
    else:
        raise exceptions.PlotlyError('Colorscale must be a valid Plotly Colorscale.The available colorscale names are {}'.format(clrs.PLOTLY_SCALES.keys()))
    values = np.linspace(0, 1, ncontours)
    vals_cmap = np.array([pair[0] for pair in cmap])
    cols = np.array([pair[1] for pair in cmap])
    inds = np.searchsorted(vals_cmap, values)
    if '#' in cols[0]:
        cols = [clrs.label_rgb(clrs.hex_to_rgb(col)) for col in cols]
    colors = [cols[0]]
    for ind, val in zip(inds[1:], values[1:]):
        val1, val2 = (vals_cmap[ind - 1], vals_cmap[ind])
        interm = (val - val1) / (val2 - val1)
        col = clrs.find_intermediate_color(cols[ind - 1], cols[ind], interm, colortype='rgb')
        colors.append(col)
    return colors