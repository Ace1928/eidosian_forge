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
def _retrieve_discovery_doc(url, http, cache_discovery, serviceName, version, cache=None, developerKey=None, num_retries=1, static_discovery=None):
    """Retrieves the discovery_doc from cache or the internet.

  Args:
    url: string, the URL of the discovery document.
    http: httplib2.Http, An instance of httplib2.Http or something that acts
      like it through which HTTP requests will be made.
    cache_discovery: Boolean, whether or not to cache the discovery doc.
    serviceName: string, name of the service.
    version: string, the version of the service.
    cache: googleapiclient.discovery_cache.base.Cache, an optional cache
      object for the discovery documents.
    developerKey: string, Key for controlling API usage, generated
      from the API Console.
    num_retries: Integer, number of times to retry discovery with
      randomized exponential backoff in case of intermittent/connection issues.
    static_discovery: Boolean, whether or not to use the static discovery docs
      included in the library.

  Returns:
    A unicode string representation of the discovery document.
  """
    from . import discovery_cache
    if cache_discovery:
        if cache is None:
            cache = discovery_cache.autodetect()
        if cache:
            content = cache.get(url)
            if content:
                return content
    if static_discovery is True or static_discovery is None:
        content = discovery_cache.get_static_doc(serviceName, version)
        if content:
            return content
        else:
            logger.warn('Failed to find discovery document from static ' + "artifacts for '%s.%s'. Please manually include the " + 'discovery document to your binary, more details are in ' + 'go/python-static-artifact-flip.', serviceName, version)
    actual_url = url
    if 'REMOTE_ADDR' in os.environ:
        actual_url = _add_query_parameter(url, 'userIp', os.environ['REMOTE_ADDR'])
    if developerKey:
        actual_url = _add_query_parameter(url, 'key', developerKey)
    logger.debug('URL being requested: GET %s', actual_url)
    req = HttpRequest(http, HttpRequest.null_postproc, actual_url)
    resp, content = req.execute(num_retries=num_retries)
    try:
        content = content.decode('utf-8')
    except AttributeError:
        pass
    try:
        service = json.loads(content)
    except ValueError as e:
        logger.error('Failed to parse as JSON: ' + content)
        raise InvalidJsonError()
    if cache_discovery and cache:
        cache.set(url, content)
    return content