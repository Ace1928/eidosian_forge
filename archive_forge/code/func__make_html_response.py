import collections
import gc
import io
import ipaddress
import json
import platform
import socket
import sys
import traceback
from debtcollector import removals
import jinja2
from oslo_utils import reflection
from oslo_utils import strutils
from oslo_utils import timeutils
import stevedore
import webob.dec
import webob.exc
import webob.response
from oslo_middleware import base
from oslo_middleware.healthcheck import opts
def _make_html_response(self, results, healthy):
    try:
        hostname = socket.gethostname()
    except socket.error:
        hostname = None
    translated_results = []
    for result in results:
        translated_results.append({'details': result.details or '', 'reason': result.reason, 'class': reflection.get_class_name(result, fully_qualified=False)})
    params = {'healthy': healthy, 'hostname': hostname, 'results': translated_results, 'detailed': self._show_details, 'now': str(timeutils.utcnow()), 'python_version': sys.version, 'platform': platform.platform(), 'gc': {'counts': gc.get_count(), 'threshold': gc.get_threshold()}, 'threads': self._get_threadstacks(), 'greenthreads': self._get_threadstacks()}
    body = _expand_template(self.HTML_RESPONSE_TEMPLATE, params)
    return (body.strip(), 'text/html')