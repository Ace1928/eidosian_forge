from __future__ import absolute_import
import io
import json
import os
import sys
import time
import webbrowser
from gcs_oauth2_boto_plugin import oauth2_client
import oauth2client.client
from six.moves import input  # pylint: disable=redefined-builtin
def SetFallbackClientIdAndSecret(client_id, client_secret):
    global CLIENT_ID
    global CLIENT_SECRET
    CLIENT_ID = client_id
    CLIENT_SECRET = client_secret