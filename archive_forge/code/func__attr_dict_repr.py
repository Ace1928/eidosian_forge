from ipywidgets import register, widget_serialization
from traitlets import validate, TraitError, Undefined
from ipydatawidgets import NDArrayWidget, get_union_array
from .Geometry import _make_key_filter
from .BufferGeometry_autogen import BufferGeometry as BufferGeometryBase
def _attr_dict_repr(d):
    parts = []
    for key, value in d.items():
        if isinstance(value, tuple):
            value_parts = [_attr_value_repr(v) for v in value]
        else:
            value_parts = [_attr_value_repr(value)]
        parts.append('%r: %s' % (key, ', '.join(value_parts)))
    return '{%s}' % (', '.join(parts),)