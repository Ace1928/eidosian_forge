from ipywidgets import register, widget_serialization
from traitlets import validate, TraitError, Undefined
from ipydatawidgets import NDArrayWidget, get_union_array
from .Geometry import _make_key_filter
from .BufferGeometry_autogen import BufferGeometry as BufferGeometryBase
def _dict_is_default(ht, name):
    value = getattr(ht, name)
    return getattr(ht.__class__, name).default_value == Undefined and (value is None or len(value) == 0)