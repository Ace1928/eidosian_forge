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
def _fix_up_method_description(method_desc, root_desc, schema):
    """Updates a method description in a discovery document.

  SIDE EFFECTS: Changes the parameters dictionary in the method description with
  extra parameters which are used locally.

  Args:
    method_desc: Dictionary with metadata describing an API method. Value comes
        from the dictionary of methods stored in the 'methods' key in the
        deserialized discovery document.
    root_desc: Dictionary; the entire original deserialized discovery document.
    schema: Object, mapping of schema names to schema descriptions.

  Returns:
    Tuple (path_url, http_method, method_id, accept, max_size, media_path_url)
    where:
      - path_url is a String; the relative URL for the API method. Relative to
        the API root, which is specified in the discovery document.
      - http_method is a String; the HTTP method used to call the API method
        described in the method description.
      - method_id is a String; the name of the RPC method associated with the
        API method, and is in the method description in the 'id' key.
      - accept is a list of strings representing what content types are
        accepted for media upload. Defaults to empty list if not in the
        discovery document.
      - max_size is a long representing the max size in bytes allowed for a
        media upload. Defaults to 0L if not in the discovery document.
      - media_path_url is a String; the absolute URI for media upload for the
        API method. Constructed using the API root URI and service path from
        the discovery document and the relative path for the API method. If
        media upload is not supported, this is None.
  """
    path_url = method_desc['path']
    http_method = method_desc['httpMethod']
    method_id = method_desc['id']
    parameters = _fix_up_parameters(method_desc, root_desc, http_method, schema)
    accept, max_size, media_path_url = _fix_up_media_upload(method_desc, root_desc, path_url, parameters)
    return (path_url, http_method, method_id, accept, max_size, media_path_url)