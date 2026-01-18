from __future__ import absolute_import
import sys
import types
from sentry_sdk._functools import wraps
from sentry_sdk.hub import Hub
from sentry_sdk._compat import reraise
from sentry_sdk.utils import capture_internal_exceptions, event_from_exception
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
class BeamIntegration(Integration):
    identifier = 'beam'

    @staticmethod
    def setup_once():
        from apache_beam.transforms.core import DoFn, ParDo
        ignore_logger('root')
        ignore_logger('bundle_processor.create')
        function_patches = ['process', 'start_bundle', 'finish_bundle', 'setup']
        for func_name in function_patches:
            setattr(DoFn, INSPECT_FUNC.format(func_name), _wrap_inspect_call(DoFn, func_name))
        old_init = ParDo.__init__

        def sentry_init_pardo(self, fn, *args, **kwargs):
            if not getattr(self, '_sentry_is_patched', False):
                for func_name in function_patches:
                    if not hasattr(fn, func_name):
                        continue
                    wrapped_func = WRAPPED_FUNC.format(func_name)
                    process_func = getattr(fn, func_name)
                    inspect_func = getattr(fn, INSPECT_FUNC.format(func_name))
                    if not getattr(inspect_func, USED_FUNC, False) and (not getattr(process_func, USED_FUNC, False)):
                        setattr(fn, wrapped_func, process_func)
                        setattr(fn, func_name, _wrap_task_call(process_func))
                self._sentry_is_patched = True
            old_init(self, fn, *args, **kwargs)
        ParDo.__init__ = sentry_init_pardo