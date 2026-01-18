from django.db import NotSupportedError
from django.db.models import Func, Index
from django.utils.functional import cached_property
class GistIndex(PostgresIndex):
    suffix = 'gist'

    def __init__(self, *expressions, buffering=None, fillfactor=None, **kwargs):
        self.buffering = buffering
        self.fillfactor = fillfactor
        super().__init__(*expressions, **kwargs)

    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        if self.buffering is not None:
            kwargs['buffering'] = self.buffering
        if self.fillfactor is not None:
            kwargs['fillfactor'] = self.fillfactor
        return (path, args, kwargs)

    def get_with_params(self):
        with_params = []
        if self.buffering is not None:
            with_params.append('buffering = %s' % ('on' if self.buffering else 'off'))
        if self.fillfactor is not None:
            with_params.append('fillfactor = %d' % self.fillfactor)
        return with_params