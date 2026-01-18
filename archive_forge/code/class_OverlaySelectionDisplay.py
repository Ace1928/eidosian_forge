from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
class OverlaySelectionDisplay(SelectionDisplay):
    """
    Selection display base class that represents selections by overlaying
    colored subsets on top of the original element in an Overlay container.
    """

    def __init__(self, color_prop='color', is_cmap=False, supports_region=True):
        if not isinstance(color_prop, (list, tuple)):
            self.color_props = [color_prop]
        else:
            self.color_props = color_prop
        self.is_cmap = is_cmap
        self.supports_region = supports_region

    def _get_color_kwarg(self, color):
        return {color_prop: [color] if self.is_cmap else color for color_prop in self.color_props}

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None, cache=None):
        from .element import Histogram
        num_layers = len(selection_streams.style_stream.colors)
        if not num_layers:
            return Overlay()
        layers = []
        for layer_number in range(num_layers):
            streams = [selection_streams.exprs_stream]
            obj = hvobj.clone(link=False) if layer_number == 1 else hvobj
            cmap_stream = selection_streams.cmap_streams[layer_number]
            layer = obj.apply(self._build_layer_callback, streams=[cmap_stream] + streams, layer_number=layer_number, cache=cache, per_element=True)
            layers.append(layer)
        for layer_number in range(num_layers):
            layer = layers[layer_number]
            cmap_stream = selection_streams.cmap_streams[layer_number]
            streams = [selection_streams.style_stream, cmap_stream]
            layer = layer.apply(self._apply_style_callback, layer_number=layer_number, streams=streams, per_element=True)
            layers[layer_number] = layer
        if region_stream is not None and self.supports_region:

            def update_region(element, region_element, colors, **kwargs):
                unselected_color = colors[0]
                if region_element is None:
                    region_element = element._empty_region()
                return self._style_region_element(region_element, unselected_color)
            streams = [region_stream, selection_streams.style_stream]
            region = hvobj.clone(link=False).apply(update_region, streams, link_dataset=False)
            eltype = hvobj.type if isinstance(hvobj, DynamicMap) else type(hvobj)
            if getattr(eltype, '_selection_dims', None) == 1 or issubclass(eltype, Histogram):
                layers.insert(1, region)
            else:
                layers.append(region)
        return Overlay(layers).collate()

    @classmethod
    def _inject_cmap_in_pipeline(cls, pipeline, cmap):
        operations = []
        for op in pipeline.operations:
            if hasattr(op, 'cmap'):
                op = op.instance(cmap=cmap)
            operations.append(op)
        return pipeline.instance(operations=operations)

    def _build_layer_callback(self, element, exprs, layer_number, cmap, cache, **kwargs):
        selection = self._select(element, exprs[layer_number], cache)
        pipeline = element.pipeline
        if cmap is not None:
            pipeline = self._inject_cmap_in_pipeline(pipeline, cmap)
        if element is selection:
            return pipeline(element.dataset)
        else:
            return pipeline(selection)

    def _apply_style_callback(self, element, layer_number, colors, cmap, alpha, **kwargs):
        opts = {}
        if layer_number == 0:
            opts['colorbar'] = False
        else:
            alpha = 1
        if cmap is not None:
            opts['cmap'] = cmap
        color = colors[layer_number] if colors else None
        return self._build_element_layer(element, color, alpha, **opts)

    def _build_element_layer(self, element, layer_color, layer_alpha, selection_expr=True):
        raise NotImplementedError()

    def _style_region_element(self, region_element, unselected_cmap):
        raise NotImplementedError()