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
def _ProbeObjectAccessWithClient(self, key, use_service_account, provider, client_email, gcs_path, generation, logger, region, billing_project):
    """Performs a head request against a signed URL to check for read access."""
    signed_url = _GenSignedUrl(key=key, api=self.gsutil_api, use_service_account=use_service_account, provider=provider, client_id=client_email, method='HEAD', duration=timedelta(seconds=60), gcs_path=gcs_path, generation=generation, logger=logger, region=region, billing_project=billing_project, string_to_sign_debug=True)
    try:
        h = GetNewHttp()
        req = Request(signed_url, 'HEAD')
        response = MakeRequest(h, req)
        if response.status_code not in [200, 403, 404]:
            raise HttpError.FromResponse(response)
        return response.status_code
    except HttpError as http_error:
        if http_error.has_attr('response'):
            error_response = http_error.response
            error_string = 'Unexpected HTTP response code %s while querying object readability. Is your system clock accurate?' % error_response.status_code
            if error_response.content:
                error_string += ' Content: %s' % error_response.content
        else:
            error_string = 'Expected an HTTP response code of 200 while querying object readability, but received an error: %s' % http_error
        raise CommandException(error_string)