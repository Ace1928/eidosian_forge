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
def _CheckAndGetCredentials(logger):
    """Returns credentials from the configuration file, if any are present.

  Args:
    logger: logging.Logger instance for outputting messages.

  Returns:
    OAuth2Credentials object if any valid ones are found, otherwise None.
  """
    configured_cred_types = []
    failed_cred_type = None
    try:
        if _HasOauth2UserAccountCreds():
            configured_cred_types.append(CredTypes.OAUTH2_USER_ACCOUNT)
        if _HasOauth2ServiceAccountCreds():
            configured_cred_types.append(CredTypes.OAUTH2_SERVICE_ACCOUNT)
        if len(configured_cred_types) > 1:
            raise CommandException('You have multiple types of configured credentials (%s), which is not supported. One common way this happens is if you run gsutil config to create credentials and later run gcloud auth, and create a second set of credentials. Your boto config path is: %s. For more help, see "gsutil help creds".' % (configured_cred_types, GetFriendlyConfigFilePaths()))
        failed_cred_type = CredTypes.OAUTH2_USER_ACCOUNT
        user_creds = _GetOauth2UserAccountCredentials()
        failed_cred_type = CredTypes.OAUTH2_SERVICE_ACCOUNT
        service_account_creds = _GetOauth2ServiceAccountCredentials()
        failed_cred_type = CredTypes.EXTERNAL_ACCOUNT
        external_account_creds = _GetExternalAccountCredentials()
        failed_cred_type = CredTypes.EXTERNAL_ACCOUNT_AUTHORIZED_USER
        external_account_authorized_user_creds = _GetExternalAccountAuthorizedUserCredentials()
        failed_cred_type = CredTypes.GCE
        gce_creds = _GetGceCreds()
        failed_cred_type = CredTypes.DEVSHELL
        devshell_creds = _GetDevshellCreds()
        creds = user_creds or service_account_creds or gce_creds or external_account_creds or external_account_authorized_user_creds or devshell_creds
        if _HasImpersonateServiceAccount() and creds:
            failed_cred_type = CredTypes.IMPERSONATION
            return _GetImpersonationCredentials(creds, logger)
        else:
            return creds
    except Exception as e:
        if failed_cred_type:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(traceback.format_exc())
            if failed_cred_type == CredTypes.IMPERSONATION:
                raise e
            elif system_util.InvokedViaCloudSdk():
                logger.warn('Your "%s" credentials are invalid. Please run\n  $ gcloud auth login', failed_cred_type)
            else:
                logger.warn('Your "%s" credentials are invalid. For more help, see "gsutil help creds", or re-run the gsutil config command (see "gsutil help config").', failed_cred_type)
        raise