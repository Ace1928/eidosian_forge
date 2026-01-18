from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import sys
import textwrap
from googlecloudsdk.api_lib.auth import exceptions as auth_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from oauth2client import client
import six
def _ValidateHost(host):
    if host in credentialed_domains:
        return True
    for suffix in credentialed_domains_suffix:
        if host.endswith(suffix):
            return True
    return False