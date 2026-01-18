from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
import six
class HttpRequestError(RequestError):
    """Raised for generic HTTP errors.

  Used for HTTP requests sent to an endpoint other than the Apigee Management
  API.
  """

    def __init__(self, status_code, reason, content):
        err_tmpl = '- HTTP status: {}\n- Reason: {}\n- Message: {}\nUse the --log-http flag to see more information.'
        self.message = err_tmpl.format(status_code, reason, content)
        super(HttpRequestError, self).__init__(message=self.message)