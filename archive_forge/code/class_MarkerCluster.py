import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class MarkerCluster(Layer):
    """MarkerCluster class, with Layer as parent class.

    A cluster of markers that you can put on the map like other layers.

    Attributes
    ----------
    markers: list, default []
        List of markers to include in the cluster.
    """
    _view_name = Unicode('LeafletMarkerClusterView').tag(sync=True)
    _model_name = Unicode('LeafletMarkerClusterModel').tag(sync=True)
    markers = Tuple().tag(trait=Instance(Layer), sync=True, **widget_serialization)
    disable_clustering_at_zoom = Int(18).tag(sync=True, o=True)
    max_cluster_radius = Int(80).tag(sync=True, o=True)