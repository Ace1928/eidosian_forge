import plotly.colors as clrs
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
from plotly.validators.heatmap import ColorscaleValidator

        Get annotations for each cell of the heatmap with graph_objs.Annotation

        :rtype (list[dict]) annotations: list of annotations for each cell of
            the heatmap
        