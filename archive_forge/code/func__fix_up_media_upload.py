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
def _fix_up_media_upload(method_desc, root_desc, path_url, parameters):
    """Adds 'media_body' and 'media_mime_type' parameters if supported by method.

  SIDE EFFECTS: If there is a 'mediaUpload' in the method description, adds
  'media_upload' key to parameters.

  Args:
    method_desc: Dictionary with metadata describing an API method. Value comes
        from the dictionary of methods stored in the 'methods' key in the
        deserialized discovery document.
    root_desc: Dictionary; the entire original deserialized discovery document.
    path_url: String; the relative URL for the API method. Relative to the API
        root, which is specified in the discovery document.
    parameters: A dictionary describing method parameters for method described
        in method_desc.

  Returns:
    Triple (accept, max_size, media_path_url) where:
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
    media_upload = method_desc.get('mediaUpload', {})
    accept = media_upload.get('accept', [])
    max_size = _media_size_to_long(media_upload.get('maxSize', ''))
    media_path_url = None
    if media_upload:
        media_path_url = _media_path_url_from_info(root_desc, path_url)
        parameters['media_body'] = MEDIA_BODY_PARAMETER_DEFAULT_VALUE.copy()
        parameters['media_mime_type'] = MEDIA_MIME_TYPE_PARAMETER_DEFAULT_VALUE.copy()
    return (accept, max_size, media_path_url)