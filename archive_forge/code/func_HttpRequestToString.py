import gzip
import hashlib
import io
import logging
import os
import re
import socket
import sys
import time
import urllib
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine._internal import six_subset
def HttpRequestToString(req, include_data=True):
    """Converts a urllib2.Request to a string.

  Args:
    req: urllib2.Request
  Returns:
    Multi-line string representing the request.
  """
    headers = ''
    for header in req.header_items():
        headers += '%s: %s\n' % (header[0], header[1])
    template = '%(method)s %(selector)s %(type)s/1.1\nHost: %(host)s\n%(headers)s'
    if include_data:
        template = template + '\n%(data)s'
    req_selector = req.selector if hasattr(req, 'selector') else req.get_selector
    if req_selector is None:
        req_selector = ''
    req_type = req.type if hasattr(req, 'type') else req.get_type()
    if req_type is None:
        req_type = ''
    req_host = req.host if hasattr(req, 'host') else req.get_host()
    if req_host is None:
        req_host = ''
    req_data = req.data if hasattr(req, 'data') else req.get_data()
    if req_data is None:
        req_data = ''
    return template % {'method': req.get_method(), 'selector': req_selector, 'type': req_type.upper(), 'host': req_host, 'headers': headers, 'data': req_data}