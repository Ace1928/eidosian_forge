import enum
import warnings
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import Promise
from django.utils.version import PY311, PY312
class ChoicesType(EnumType):
    """A metaclass for creating a enum choices."""

    def __new__(metacls, classname, bases, classdict, **kwds):
        labels = []
        for key in classdict._member_names:
            value = classdict[key]
            if isinstance(value, (list, tuple)) and len(value) > 1 and isinstance(value[-1], (Promise, str)):
                *value, label = value
                value = tuple(value)
            else:
                label = key.replace('_', ' ').title()
            labels.append(label)
            dict.__setitem__(classdict, key, value)
        cls = super().__new__(metacls, classname, bases, classdict, **kwds)
        for member, label in zip(cls.__members__.values(), labels):
            member._label_ = label
        return enum.unique(cls)
    if not PY312:

        def __contains__(cls, member):
            if not isinstance(member, enum.Enum):
                return any((x.value == member for x in cls))
            return super().__contains__(member)

    @property
    def names(cls):
        empty = ['__empty__'] if hasattr(cls, '__empty__') else []
        return empty + [member.name for member in cls]

    @property
    def choices(cls):
        empty = [(None, cls.__empty__)] if hasattr(cls, '__empty__') else []
        return empty + [(member.value, member.label) for member in cls]

    @property
    def labels(cls):
        return [label for _, label in cls.choices]

    @property
    def values(cls):
        return [value for value, _ in cls.choices]