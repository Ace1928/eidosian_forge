from __future__ import absolute_import
import six
from six.moves import zip
from six import BytesIO
from six.moves import http_client
from six.moves.urllib.parse import urlencode, urlparse, urljoin, urlunparse, parse_qsl
import copy
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import json
import keyword
import logging
import mimetypes
import os
import re
import httplib2
import uritemplate
import google.api_core.client_options
from google.auth.transport import mtls
from google.auth.exceptions import MutualTLSChannelError
from googleapiclient import _auth
from googleapiclient import mimeparse
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidJsonError
from googleapiclient.errors import MediaUploadSizeError
from googleapiclient.errors import UnacceptableMimeTypeError
from googleapiclient.errors import UnknownApiNameOrVersion
from googleapiclient.errors import UnknownFileType
from googleapiclient.http import build_http
from googleapiclient.http import BatchHttpRequest
from googleapiclient.http import HttpMock
from googleapiclient.http import HttpMockSequence
from googleapiclient.http import HttpRequest
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaUpload
from googleapiclient.model import JsonModel
from googleapiclient.model import MediaModel
from googleapiclient.model import RawModel
from googleapiclient.schema import Schemas
from googleapiclient._helpers import _add_query_parameter
from googleapiclient._helpers import positional
def _fix_up_parameters(method_desc, root_desc, http_method, schema):
    """Updates parameters of an API method with values specific to this library.

  Specifically, adds whatever global parameters are specified by the API to the
  parameters for the individual method. Also adds parameters which don't
  appear in the discovery document, but are available to all discovery based
  APIs (these are listed in STACK_QUERY_PARAMETERS).

  SIDE EFFECTS: This updates the parameters dictionary object in the method
  description.

  Args:
    method_desc: Dictionary with metadata describing an API method. Value comes
        from the dictionary of methods stored in the 'methods' key in the
        deserialized discovery document.
    root_desc: Dictionary; the entire original deserialized discovery document.
    http_method: String; the HTTP method used to call the API method described
        in method_desc.
    schema: Object, mapping of schema names to schema descriptions.

  Returns:
    The updated Dictionary stored in the 'parameters' key of the method
        description dictionary.
  """
    parameters = method_desc.setdefault('parameters', {})
    for name, description in six.iteritems(root_desc.get('parameters', {})):
        parameters[name] = description
    for name in STACK_QUERY_PARAMETERS:
        parameters[name] = STACK_QUERY_PARAMETER_DEFAULT_VALUE.copy()
    if http_method in HTTP_PAYLOAD_METHODS and 'request' in method_desc:
        body = BODY_PARAMETER_DEFAULT_VALUE.copy()
        body.update(method_desc['request'])
        parameters['body'] = body
    return parameters