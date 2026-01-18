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
def Check(p):
    """Warn about other credential helpers that will be ignored."""
    if not os.path.exists(p):
        return
    try:
        data = files.ReadFileContents(p)
        if 'source.developers.google.com' in data:
            sys.stderr.write(textwrap.dedent("You have credentials for your Google repository in [{path}]. This repository's\ngit credential helper is set correctly, so the credentials in [{path}] will not\nbe used, but you may want to remove them to avoid confusion.\n".format(path=p)))
    except Exception:
        pass