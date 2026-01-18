from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import json
import logging
import os
import io
import six
import traceback
from apitools.base.py import credentials_lib
from apitools.base.py import exceptions as apitools_exceptions
from boto import config
from gslib.cred_types import CredTypes
from gslib.exception import CommandException
from gslib.impersonation_credentials import ImpersonationCredentials
from gslib.no_op_credentials import NoOpCredentials
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils.boto_util import GetFriendlyConfigFilePaths
from gslib.utils.boto_util import GetCredentialStoreFilename
from gslib.utils.boto_util import GetGceCredentialCacheFilename
from gslib.utils.boto_util import GetGcsJsonApiVersion
from gslib.utils.constants import UTF8
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from oauth2client.client import HAS_CRYPTO
from oauth2client.contrib import devshell
from oauth2client.service_account import ServiceAccountCredentials
from google_reauth import reauth_creds
from oauth2client.contrib import multiprocess_file_storage
from six import BytesIO
def _GetExternalAccountAuthorizedUserCredentials():
    external_account_authorized_user_filename = config.get('Credentials', 'gs_external_account_authorized_user_file', None)
    if not external_account_authorized_user_filename:
        return None
    return WrappedCredentials.for_external_account_authorized_user(external_account_authorized_user_filename)