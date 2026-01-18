import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
def _parse_backend_opt(self, opt, plot, model_accessor_aliases):
    """
        Parses a custom option of the form 'model.accessor.option'
        and returns the corresponding model and accessor.
        """
    accessors = opt.split('.')
    if len(accessors) < 2:
        self.param.warning(f"Custom option {opt!r} expects at least two accessors separated by '.'")
        return
    model_accessor = accessors[0]
    model_accessor = model_accessor_aliases.get(model_accessor) or model_accessor
    if model_accessor in self.handles:
        model = self.handles[model_accessor]
    elif hasattr(plot, model_accessor):
        model = getattr(plot, model_accessor)
    else:
        self.param.warning(f'{model_accessor} model could not be resolved on {type(self).__name__!r} plot. Ensure the {opt!r} custom option spec references a valid model in the plot.handles {list(self.handles.keys())!r} or on the underlying figure object.')
        return
    for acc in accessors[1:-1]:
        if '[' in acc and acc.endswith(']'):
            getitem_index = acc.index('[')
            getitem_spec = acc[getitem_index + 1:-1]
            try:
                if ':' in getitem_spec:
                    slice_parts = getitem_spec.split(':')
                    slice_start = None if slice_parts[0] == '' else int(slice_parts[0])
                    slice_stop = None if slice_parts[1] == '' else int(slice_parts[1])
                    slice_step = None if len(slice_parts) < 3 or slice_parts[2] == '' else int(slice_parts[2])
                    getitem_acc = slice(slice_start, slice_stop, slice_step)
                elif ',' in getitem_spec:
                    getitem_acc = [literal_eval(item.strip()) for item in getitem_spec.split(',')]
                else:
                    getitem_acc = literal_eval(getitem_spec)
            except Exception:
                self.param.warning(f'Could not evaluate getitem {getitem_spec!r} in custom option spec {opt!r}.')
                model = None
                break
            acc = acc[:getitem_index]
        else:
            getitem_acc = None
        if '(' in acc and ')' in acc:
            method_ini_index = acc.index('(')
            method_end_index = acc.index(')')
            method_spec = acc[method_ini_index + 1:method_end_index]
            try:
                if method_spec:
                    method_parts = method_spec.split(',')
                    method_args = []
                    method_kwargs = {}
                    for part in method_parts:
                        if '=' in part:
                            key, value = part.split('=')
                            method_kwargs[key.strip()] = literal_eval(value.strip())
                        else:
                            method_args.append(literal_eval(part.strip()))
                else:
                    method_args = ()
                    method_kwargs = {}
            except Exception:
                self.param.warning(f'Could not evaluate method arguments {method_spec!r} in custom option spec {opt!r}.')
                model = None
                break
            acc = acc[:method_ini_index]
            if not isinstance(model, list):
                model = getattr(model, acc)(*method_args, **method_kwargs)
            else:
                model = [getattr(m, acc)(*method_args, **method_kwargs) for m in model]
            if getitem_acc is not None:
                if not isinstance(getitem_acc, list):
                    model = model.__getitem__(getitem_acc)
                else:
                    model = [model.__getitem__(i) for i in getitem_acc]
            acc = acc[method_end_index:]
        if acc == '' or model is None:
            continue
        if not hasattr(model, acc):
            self.param.warning(f'Could not resolve {acc!r} attribute on {type(model).__name__!r} model. Ensure the custom option spec you provided references a valid submodel.')
            model = None
            break
        model = getattr(model, acc)
    attr_accessor = accessors[-1]
    return (model, attr_accessor)