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
@bothmethod
def _install_param_callbacks(self_or_cls, inst):

    def update_selection_mode(*_):
        for stream in inst._selection_expr_streams.values():
            stream.reset()
            stream.mode = inst.selection_mode
    inst.param.watch(update_selection_mode, ['selection_mode'])

    def update_cross_filter_mode(*_):
        inst._cross_filter_stream.reset()
        inst._cross_filter_stream.mode = inst.cross_filter_mode
    inst.param.watch(update_cross_filter_mode, ['cross_filter_mode'])

    def update_show_region(*_):
        for stream in inst._selection_expr_streams.values():
            stream.include_region = inst.show_regions
            stream.event()
    inst.param.watch(update_show_region, ['show_regions'])

    def update_selection_expr(*_):
        new_selection_expr = inst.selection_expr
        current_selection_expr = inst._cross_filter_stream.selection_expr
        if repr(new_selection_expr) != repr(current_selection_expr):
            if inst.show_regions:
                inst.show_regions = False
            inst._selection_override.event(selection_expr=new_selection_expr)
    inst.param.watch(update_selection_expr, ['selection_expr'])

    def selection_expr_changed(*_):
        new_selection_expr = inst._cross_filter_stream.selection_expr
        if repr(inst.selection_expr) != repr(new_selection_expr):
            inst.selection_expr = new_selection_expr
    inst._cross_filter_stream.param.watch(selection_expr_changed, ['selection_expr'])
    for stream in inst._selection_expr_streams.values():

        def clear_stream_history(resetting, stream=stream):
            if resetting:
                stream.clear_history()
        print('registering reset for ', stream)
        stream.plot_reset_stream.param.watch(clear_stream_history, ['resetting'])