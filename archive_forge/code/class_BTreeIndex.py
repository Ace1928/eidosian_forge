from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class BTreeIndex(PostgresIndex):
    suffix = 'btree'

    def __init__(self, *expressions, fillfactor=None, **kwargs):
        self.fillfactor = fillfactor
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.fillfactor is not None:
            kwargs['fillfactor'] = self.fillfactor
        return (path, args, kwargs)

    def get_with_params(self):
        with_params = []
        if self.fillfactor is not None:
            with_params.append('fillfactor = %d' % self.fillfactor)
        return with_params