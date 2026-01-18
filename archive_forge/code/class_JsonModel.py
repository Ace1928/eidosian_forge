from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
class JsonModel(BaseModel):
    """Model class for JSON.

  Serializes and de-serializes between JSON and the Python
  object representation of HTTP request and response bodies.
  """
    accept = 'application/json'
    content_type = 'application/json'
    alt_param = 'json'

    def __init__(self, data_wrapper=False):
        """Construct a JsonModel.

    Args:
      data_wrapper: boolean, wrap requests and responses in a data wrapper
    """
        self._data_wrapper = data_wrapper

    def serialize(self, body_value):
        if isinstance(body_value, dict) and 'data' not in body_value and self._data_wrapper:
            body_value = {'data': body_value}
        return json.dumps(body_value)

    def deserialize(self, content):
        try:
            content = content.decode('utf-8')
        except AttributeError:
            pass
        body = json.loads(content)
        if self._data_wrapper and isinstance(body, dict) and ('data' in body):
            body = body['data']
        return body

    @property
    def no_content_response(self):
        return {}