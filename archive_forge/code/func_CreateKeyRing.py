from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from boto import config
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.gcs_json_credentials import SetUpJsonCredentialsAndCache
from gslib.no_op_credentials import NoOpCredentials
from gslib.third_party.kms_apitools import cloudkms_v1_client as apitools_client
from gslib.third_party.kms_apitools import cloudkms_v1_messages as apitools_messages
from gslib.utils import system_util
from gslib.utils.boto_util import GetCertsFile
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.boto_util import GetNumRetries
def CreateKeyRing(self, project, keyring_name, location='global'):
    """Attempts to create the specified keyRing.

    Args:
      project: (str) The project id in which to create the keyRing and key.
      keyring_name: (str) The name of the keyRing, e.g. my-keyring. Note
          that this must be unique within the location.
      location: (str) The location in which to create the keyRing. Defaults to
          'global'.

    Returns:
      (str) The fully-qualified name of the keyRing, e.g.:
      projects/my-project/locations/global/keyRings/my-keyring

    Raises:
      Translated CloudApi exception if we were unable to create the keyRing.
      Note that in the event of a 409 status code (resource already exists) when
      attempting creation, we continue and treat this as a success.
    """
    keyring_msg = apitools_messages.KeyRing(name='projects/%s/locations/%s/keyRings/%s' % (project, location, keyring_name))
    keyring_create_request = apitools_messages.CloudkmsProjectsLocationsKeyRingsCreateRequest(keyRing=keyring_msg, keyRingId=keyring_name, parent='projects/%s/locations/%s' % (project, location))
    try:
        self.api_client.projects_locations_keyRings.Create(keyring_create_request)
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        if e.status_code != 409:
            raise
    return 'projects/%s/locations/%s/keyRings/%s' % (project, location, keyring_name)