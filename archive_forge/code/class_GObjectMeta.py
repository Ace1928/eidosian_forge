import re
from ._constants import TYPE_INVALID
from .docstring import generate_doc_string
from ._gi import \
from . import _gi
from . import _propertyhelper as propertyhelper
from . import _signalhelper as signalhelper
class GObjectMeta(_GObjectMetaBase, MetaClassHelper):
    """Meta class used for GI GObject based types."""

    def __init__(cls, name, bases, dict_):
        super(GObjectMeta, cls).__init__(name, bases, dict_)
        is_gi_defined = False
        if cls.__module__ == 'gi.repository.' + cls.__info__.get_namespace():
            is_gi_defined = True
        is_python_defined = False
        if not is_gi_defined and cls.__module__ != GObjectMeta.__module__:
            is_python_defined = True
        if is_python_defined:
            cls._setup_vfuncs()
        elif is_gi_defined:
            if isinstance(cls.__info__, ObjectInfo):
                cls._setup_class_methods()
            cls._setup_methods()
            cls._setup_constants()
            cls._setup_native_vfuncs()
            if isinstance(cls.__info__, ObjectInfo):
                cls._setup_fields()
            elif isinstance(cls.__info__, InterfaceInfo):
                register_interface_info(cls.__info__.get_g_type())

    def mro(cls):
        return mro(cls)

    @property
    def __doc__(cls):
        """Meta class property which shows up on any class using this meta-class."""
        if cls == GObjectMeta:
            return ''
        doc = cls.__dict__.get('__doc__', None)
        if doc is not None:
            return doc
        if cls.__module__.startswith(('gi.repository.', 'gi.overrides')):
            return generate_doc_string(cls.__info__)
        return None