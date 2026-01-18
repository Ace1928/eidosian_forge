from __future__ import unicode_literals
class EnumObject(object):
    """
  Simple enumeration base. Design inspired by clang python bindings
  BaseEnumeration. Subclasses must provide class member _id_map.
  """
    _id_map = {}

    @classmethod
    def register_value(cls, value, obj):
        assert cls._id_map is not EnumObject._id_map
        if value in cls._id_map:
            raise ValueError('{0} value {1} already loaded'.format(cls.__name__, value))
        cls._id_map[value] = obj

    @classmethod
    def from_id(cls, qid):
        if qid in cls._id_map:
            return cls._id_map[qid]
        raise ValueError('{} is not a valid {} value'.format(qid, cls.__name__))

    @classmethod
    def assign_names(cls):
        for key, value in vars(cls).items():
            if isinstance(value, cls):
                value.name_ = key

    @classmethod
    def from_name(cls, name):
        if hasattr(cls, name):
            obj = getattr(cls, name)
            if isinstance(obj, cls):
                return obj
        raise ValueError('{} is not a valid {} enum'.format(name, cls.__name__))

    @classmethod
    def get(cls, name, default=None):
        if hasattr(cls, name):
            obj = getattr(cls, name)
            if isinstance(obj, cls):
                return obj
            raise ValueError('{} is not a valid {} enum'.format(name, cls.__name__))
        return default

    def __init__(self, value):
        self.value = value
        self.name_ = None
        self.__class__.register_value(value, self)

    @property
    def name(self):
        """Get the enumeration name of this value."""
        if self.name_ is None:
            self.__class__.assign_names()
        return self.name_

    def as_dict(self):
        return self.name

    def __repr__(self):
        return '{}.{}'.format(self.__class__.__name__, self.name)