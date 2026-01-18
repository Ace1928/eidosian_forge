import warnings
import functools
class _VersionTupleEnumMixin:

    @property
    def major(self):
        return self.value[0]

    @property
    def minor(self):
        return self.value[1]

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, int):
            return cls((value, 0))
        if value is None:
            return cls.default()
        return super()._missing_(value)

    def __str__(self):
        return f'{self.major}.{self.minor}'

    @classmethod
    def default(cls):
        return max(cls.__members__.values())

    @classmethod
    def supported_versions(cls):
        return frozenset(cls.__members__.values())