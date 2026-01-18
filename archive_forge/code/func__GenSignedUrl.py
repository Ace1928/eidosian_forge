from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
import copy
from datetime import datetime
from datetime import timedelta
import getpass
import json
import re
import sys
import six
from six.moves import urllib
from apitools.base.py.exceptions import HttpError
from apitools.base.py.http_wrapper import MakeRequest
from apitools.base.py.http_wrapper import Request
from boto import config
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import StorageUrlFromString
from gslib.utils import constants
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.shim_util import GcloudStorageMap, GcloudStorageFlag
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
def _GenSignedUrl(key, api, use_service_account, provider, client_id, method, duration, gcs_path, logger, region, content_type=None, billing_project=None, string_to_sign_debug=False, generation=None):
    """Construct a string to sign with the provided key.

  Args:
    key: The private key to use for signing the URL.
    api: The CloudApiDelegator instance
    use_service_account: If True, use the service account credentials
        instead of using the key file to sign the URL
    provider: Cloud storage provider to connect to.  If not present,
        class-wide default is used.
    client_id: Client ID signing this URL.
    method: The HTTP method to be used with the signed URL.
    duration: timedelta for which the constructed signed URL should be valid.
    gcs_path: String path to the bucket of object for signing, in the form
        'bucket' or 'bucket/object'.
    logger: logging.Logger for warning and debug output.
    region: Geographic region in which the requested resource resides.
    content_type: Optional Content-Type for the signed URL. HTTP requests using
        the URL must match this Content-Type.
    billing_project: Specify a user project to be billed for the request.
    string_to_sign_debug: If true AND logger is enabled for debug level,
        print string to sign to debug. Used to differentiate user's
        signed URL from the probing permissions-check signed URL.
    generation: If not None, specifies a version of an object for signing.

  Returns:
    The complete URL (string).
  """
    gs_host = config.get('Credentials', 'gs_host', 'storage.googleapis.com')
    signed_headers = {'host': gs_host}
    if method == 'RESUMABLE':
        method = 'POST'
        signed_headers['x-goog-resumable'] = 'start'
        if not content_type:
            logger.warn('Warning: no Content-Type header was specified with the -c flag, so uploads to the resulting Signed URL must not specify a Content-Type.')
    if content_type:
        signed_headers['content-type'] = content_type
    if use_service_account:
        final_url = api.SignUrl(provider=provider, method=method, duration=duration, path=gcs_path, generation=generation, logger=logger, region=region, signed_headers=signed_headers, string_to_sign_debug=string_to_sign_debug)
    else:
        if six.PY2:
            digest = b'RSA-SHA256'
        else:
            digest = 'RSA-SHA256'
        string_to_sign, canonical_query_string = CreatePayload(client_id=client_id, method=method, duration=duration, path=gcs_path, generation=generation, logger=logger, region=region, signed_headers=signed_headers, billing_project=billing_project, string_to_sign_debug=string_to_sign_debug)
        raw_signature = sign(key, string_to_sign, digest)
        final_url = GetFinalUrl(raw_signature, gs_host, gcs_path, canonical_query_string)
    return final_url