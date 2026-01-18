import random
from eventlet import wsgi
from eventlet.zipkin import api
from eventlet.zipkin._thrift.zipkinCore.constants import \
from eventlet.zipkin.http import \
def _patched_handle_one_response(self):
    api.init_trace_data()
    trace_id = int_or_none(self.headers.getheader(HDR_TRACE_ID))
    span_id = int_or_none(self.headers.getheader(HDR_SPAN_ID))
    parent_id = int_or_none(self.headers.getheader(HDR_PARENT_SPAN_ID))
    sampled = bool_or_none(self.headers.getheader(HDR_SAMPLED))
    if trace_id is None:
        trace_id = span_id = api.generate_trace_id()
        parent_id = None
        sampled = _sampler.sampling()
    ip, port = self.request.getsockname()[:2]
    ep = api.ZipkinDataBuilder.build_endpoint(ip, port)
    trace_data = api.TraceData(name=self.command, trace_id=trace_id, span_id=span_id, parent_id=parent_id, sampled=sampled, endpoint=ep)
    api.set_trace_data(trace_data)
    api.put_annotation(SERVER_RECV)
    api.put_key_value('http.uri', self.path)
    __original_handle_one_response__(self)
    if api.is_sample():
        api.put_annotation(SERVER_SEND)