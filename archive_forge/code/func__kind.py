import collections
from urllib import parse as parser
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler import _utils as utils
from osprofiler.drivers import base
from osprofiler import exc
def _kind(self, name):
    if 'wsgi' in name:
        return self.trace_api.SpanKind.SERVER
    elif 'db' in name or 'http' in name or 'api' in name:
        return self.trace_api.SpanKind.CLIENT
    return self.trace_api.SpanKind.INTERNAL