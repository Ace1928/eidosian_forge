from __future__ import absolute_import
from __future__ import unicode_literals
import http.client
from typing import Dict
from typing import Union
import warnings
from google.rpc import error_details_pb2
class ResumeIncomplete(Redirection):
    """Exception mapping a ``308 Resume Incomplete`` response.

    .. note:: :attr:`http.client.PERMANENT_REDIRECT` is ``308``, but Google
        APIs differ in their use of this status code.
    """
    code = 308