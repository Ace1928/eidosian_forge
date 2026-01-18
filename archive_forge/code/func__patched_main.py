from eventlet import greenthread
from eventlet.zipkin import api
def _patched_main(self, function, args, kwargs):
    if hasattr(self, 'trace_data'):
        api.set_trace_data(self.trace_data)
    __original_main__(self, function, args, kwargs)