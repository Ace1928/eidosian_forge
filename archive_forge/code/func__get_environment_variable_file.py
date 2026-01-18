import collections
import copy
import datetime
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import six
from six.moves import http_client
from six.moves import urllib
import oauth2client
from oauth2client import _helpers
from oauth2client import _pkce
from oauth2client import clientsecrets
from oauth2client import transport
def _get_environment_variable_file():
    application_default_credential_filename = os.environ.get(GOOGLE_APPLICATION_CREDENTIALS, None)
    if application_default_credential_filename:
        if os.path.isfile(application_default_credential_filename):
            return application_default_credential_filename
        else:
            raise ApplicationDefaultCredentialsError('File ' + application_default_credential_filename + ' (pointed by ' + GOOGLE_APPLICATION_CREDENTIALS + ' environment variable) does not exist!')