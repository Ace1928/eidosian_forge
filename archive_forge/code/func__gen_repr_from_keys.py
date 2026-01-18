from ipywidgets import register, widget_serialization
from traitlets import validate, TraitError, Undefined
from ipydatawidgets import NDArrayWidget, get_union_array
from .Geometry import _make_key_filter
from .BufferGeometry_autogen import BufferGeometry as BufferGeometryBase
def _gen_repr_from_keys(self, keys):
    data_keys = ('attributes', 'morphAttributes', 'index')
    class_name = self.__class__.__name__
    signature_parts = ['%s=%r' % (key, getattr(self, key)) for key in keys if key not in data_keys]
    if not self._compare(self.index, self.__class__.index.default_value) and self.index is not None:
        signature_parts.append('index=%s' % _attr_value_repr(self.index))
    for name in ('attributes', 'morphAttributes'):
        if not _dict_is_default(self, name):
            signature_parts.append('%s=%s' % (name, _attr_dict_repr(getattr(self, name))))
    return '%s(%s)' % (class_name, ', '.join(signature_parts))