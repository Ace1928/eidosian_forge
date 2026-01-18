import os
import subprocess
import sys
import platform
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing_utils import EnvironHeaders, should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _install_subprocess():
    old_popen_init = subprocess.Popen.__init__

    def sentry_patched_popen_init(self, *a, **kw):
        hub = Hub.current
        if hub.get_integration(StdlibIntegration) is None:
            return old_popen_init(self, *a, **kw)
        a = list(a)
        args = _init_argument(a, kw, 'args', 0) or []
        cwd = _init_argument(a, kw, 'cwd', 9)
        description = None
        if isinstance(args, (list, tuple)) and len(args) < 100:
            with capture_internal_exceptions():
                description = ' '.join(map(str, args))
        if description is None:
            description = safe_repr(args)
        env = None
        with hub.start_span(op=OP.SUBPROCESS, description=description) as span:
            for k, v in hub.iter_trace_propagation_headers(span):
                if env is None:
                    env = _init_argument(a, kw, 'env', 10, lambda x: dict(x or os.environ))
                env['SUBPROCESS_' + k.upper().replace('-', '_')] = v
            if cwd:
                span.set_data('subprocess.cwd', cwd)
            rv = old_popen_init(self, *a, **kw)
            span.set_tag('subprocess.pid', self.pid)
            return rv
    subprocess.Popen.__init__ = sentry_patched_popen_init
    old_popen_wait = subprocess.Popen.wait

    def sentry_patched_popen_wait(self, *a, **kw):
        hub = Hub.current
        if hub.get_integration(StdlibIntegration) is None:
            return old_popen_wait(self, *a, **kw)
        with hub.start_span(op=OP.SUBPROCESS_WAIT) as span:
            span.set_tag('subprocess.pid', self.pid)
            return old_popen_wait(self, *a, **kw)
    subprocess.Popen.wait = sentry_patched_popen_wait
    old_popen_communicate = subprocess.Popen.communicate

    def sentry_patched_popen_communicate(self, *a, **kw):
        hub = Hub.current
        if hub.get_integration(StdlibIntegration) is None:
            return old_popen_communicate(self, *a, **kw)
        with hub.start_span(op=OP.SUBPROCESS_COMMUNICATE) as span:
            span.set_tag('subprocess.pid', self.pid)
            return old_popen_communicate(self, *a, **kw)
    subprocess.Popen.communicate = sentry_patched_popen_communicate