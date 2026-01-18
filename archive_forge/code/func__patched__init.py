from eventlet import greenthread
from eventlet.zipkin import api
def _patched__init(self, parent):
    if api.is_tracing():
        self.trace_data = api.get_trace_data()
    __original_init__(self, parent)