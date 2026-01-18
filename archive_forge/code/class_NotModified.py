from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class NotModified(Redirection):
    """Exception mapping a ``304 Not Modified`` response."""
    code = http.client.NOT_MODIFIED