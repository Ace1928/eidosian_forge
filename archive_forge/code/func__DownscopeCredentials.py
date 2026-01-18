from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import os
import subprocess
import tempfile
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import retry
import six
def _DownscopeCredentials(token, access_boundary_json):
    """Downscope the given credentials to the given access boundary.

  Args:
    token: The credentials to downscope.
    access_boundary_json: The JSON-formatted access boundary.

  Returns:
    A downscopded credential with the given access-boundary.
  """
    payload = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'subject_token': token, 'options': access_boundary_json}
    cab_token_url = 'https://sts.googleapis.com/v1/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    downscope_response = requests.GetSession().post(cab_token_url, headers=headers, data=payload)
    if downscope_response.status_code != 200:
        raise ValueError('Error downscoping credentials')
    cab_token = json.loads(downscope_response.content)
    return cab_token.get('access_token', None)