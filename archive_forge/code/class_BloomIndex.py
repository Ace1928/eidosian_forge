from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class BloomIndex(PostgresIndex):
    suffix = 'bloom'

    def __init__(self, *expressions, length=None, columns=(), **kwargs):
        super().__init__(*expressions, **kwargs)
        if len(self.fields) > 32:
            raise ValueError('Bloom indexes support a maximum of 32 fields.')
        if not isinstance(columns, (list, tuple)):
            raise ValueError('BloomIndex.columns must be a list or tuple.')
        if len(columns) > len(self.fields):
            raise ValueError('BloomIndex.columns cannot have more values than fields.')
        if not all((0 < col <= 4095 for col in columns)):
            raise ValueError('BloomIndex.columns must contain integers from 1 to 4095.')
        if length is not None and (not 0 < length <= 4096):
            raise ValueError('BloomIndex.length must be None or an integer from 1 to 4096.')
        self.length = length
        self.columns = columns

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.length is not None:
            kwargs['length'] = self.length
        if self.columns:
            kwargs['columns'] = self.columns
        return (path, args, kwargs)

    def get_with_params(self):
        with_params = []
        if self.length is not None:
            with_params.append('length = %d' % self.length)
        if self.columns:
            with_params.extend(('col%d = %d' % (i, v) for i, v in enumerate(self.columns, start=1)))
        return with_params