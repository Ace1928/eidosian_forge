from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def HasConfiguredCredentials():
    """Determines if boto credential/config file exists."""
    has_goog_creds = config.has_option('Credentials', 'gs_access_key_id') and config.has_option('Credentials', 'gs_secret_access_key')
    has_amzn_creds = config.has_option('Credentials', 'aws_access_key_id') and config.has_option('Credentials', 'aws_secret_access_key')
    has_oauth_creds = config.has_option('Credentials', 'gs_oauth2_refresh_token')
    has_external_creds = config.has_option('Credentials', 'gs_external_account_file')
    has_external_account_authorized_user_creds = config.has_option('Credentials', 'gs_external_account_authorized_user_file')
    has_service_account_creds = HAS_CRYPTO and config.has_option('Credentials', 'gs_service_client_id') and config.has_option('Credentials', 'gs_service_key_file')
    if has_goog_creds or has_amzn_creds or has_oauth_creds or has_service_account_creds or has_external_creds or has_external_account_authorized_user_creds:
        return True
    valid_auth_handler = None
    try:
        valid_auth_handler = boto.auth.get_auth_handler(GSConnection.DefaultHost, config, Provider('google'), requested_capability=['s3'])
        if 'NoOpAuth' == getattr(getattr(valid_auth_handler, '__class__', None), '__name__', None):
            valid_auth_handler = None
    except NoAuthHandlerFound:
        pass
    return valid_auth_handler