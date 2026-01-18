import pprint
from abc import abstractmethod
class _MlflowObject:

    def __iter__(self):
        for prop in self._properties():
            yield (prop, self.__getattribute__(prop))

    @classmethod
    def _get_properties_helper(cls):
        return sorted([p for p in cls.__dict__ if isinstance(getattr(cls, p), property)])

    @classmethod
    def _properties(cls):
        return cls._get_properties_helper()

    @classmethod
    @abstractmethod
    def from_proto(cls, proto):
        pass

    @classmethod
    def from_dictionary(cls, the_dict):
        filtered_dict = {key: value for key, value in the_dict.items() if key in cls._properties()}
        return cls(**filtered_dict)

    def __repr__(self):
        return to_string(self)