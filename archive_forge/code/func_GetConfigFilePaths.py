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
def GetConfigFilePaths():
    """Returns a list of the path(s) to the boto config file(s) to be loaded."""
    potential_config_paths = BotoConfigLocations
    if 'AWS_CREDENTIAL_FILE' in os.environ:
        potential_config_paths.append(os.environ['AWS_CREDENTIAL_FILE'])
    aws_cred_file = os.path.join(os.path.expanduser('~'), '.aws', 'credentials')
    if os.path.isfile(aws_cred_file):
        potential_config_paths.append(aws_cred_file)
    readable_config_paths = []
    for path in potential_config_paths:
        try:
            with open(path, 'r'):
                readable_config_paths.append(path)
        except IOError:
            pass
    return readable_config_paths