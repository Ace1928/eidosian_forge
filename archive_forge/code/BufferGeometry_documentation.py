from ipywidgets import register, widget_serialization
from traitlets import validate, TraitError, Undefined
from ipydatawidgets import NDArrayWidget, get_union_array
from .Geometry import _make_key_filter
from .BufferGeometry_autogen import BufferGeometry as BufferGeometryBase
Creates a PlainBufferGeometry of another geometry.

        store_ref determines if the reference is stored after initalization.
        If it is, it will be used for future embedding.
        