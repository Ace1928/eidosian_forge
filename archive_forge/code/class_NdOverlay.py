from functools import reduce
import numpy as np
import param
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .layout import AdjointLayout, Composable, Layout, Layoutable
from .ndmapping import UniformNdMapping
from .util import dimensioned_streams, sanitize_identifier, unique_array
class NdOverlay(Overlayable, UniformNdMapping, CompositeOverlay):
    """
    An NdOverlay allows a group of NdOverlay to be overlaid together. NdOverlay can
    be indexed out of an overlay and an overlay is an iterable that iterates
    over the contained layers.
    """
    kdims = param.List(default=[Dimension('Element')], constant=True, doc='\n        List of dimensions the NdOverlay can be indexed by.')
    _deep_indexable = True

    def __init__(self, overlays=None, kdims=None, **params):
        super().__init__(overlays, kdims=kdims, **params)

    def decollate(self):
        """Packs NdOverlay of DynamicMaps into a single DynamicMap that returns an
        NdOverlay

        Decollation allows packing a NdOverlay of DynamicMaps into a single DynamicMap
        that returns an NdOverlay of simple (non-dynamic) elements. All nested streams
        are lifted to the resulting DynamicMap, and are available in the `streams`
        property.  The `callback` property of the resulting DynamicMap is a pure,
        stateless function of the stream values. To avoid stream parameter name
        conflicts, the resulting DynamicMap is configured with
        positional_stream_args=True, and the callback function accepts stream values
        as positional dict arguments.

        Returns:
            DynamicMap that returns an NdOverlay
        """
        from .decollate import decollate
        return decollate(self)