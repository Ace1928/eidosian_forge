from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
class ModelMeta(type):
    """Metaclass for all Models"""

    def __init__(cls, name, bases, dict):
        super(ModelMeta, cls).__init__(name, bases, dict)
        cls.__sub_classes__ = []
        from boto.sdb.db.manager import get_manager
        try:
            if filter(lambda b: issubclass(b, Model), bases):
                for base in bases:
                    base.__sub_classes__.append(cls)
                cls._manager = get_manager(cls)
                for key in dict.keys():
                    if isinstance(dict[key], Property):
                        property = dict[key]
                        property.__property_config__(cls, key)
                prop_names = []
                props = cls.properties()
                for prop in props:
                    if not prop.__class__.__name__.startswith('_'):
                        prop_names.append(prop.name)
                setattr(cls, '_prop_names', prop_names)
        except NameError:
            pass