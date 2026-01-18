from typing import List
from redis import DataError
class Field:
    NUMERIC = 'NUMERIC'
    TEXT = 'TEXT'
    WEIGHT = 'WEIGHT'
    GEO = 'GEO'
    TAG = 'TAG'
    VECTOR = 'VECTOR'
    SORTABLE = 'SORTABLE'
    NOINDEX = 'NOINDEX'
    AS = 'AS'
    GEOSHAPE = 'GEOSHAPE'

    def __init__(self, name: str, args: List[str]=None, sortable: bool=False, no_index: bool=False, as_name: str=None):
        if args is None:
            args = []
        self.name = name
        self.args = args
        self.args_suffix = list()
        self.as_name = as_name
        if sortable:
            self.args_suffix.append(Field.SORTABLE)
        if no_index:
            self.args_suffix.append(Field.NOINDEX)
        if no_index and (not sortable):
            raise ValueError('Non-Sortable non-Indexable fields are ignored')

    def append_arg(self, value):
        self.args.append(value)

    def redis_args(self):
        args = [self.name]
        if self.as_name:
            args += [self.AS, self.as_name]
        args += self.args
        args += self.args_suffix
        return args