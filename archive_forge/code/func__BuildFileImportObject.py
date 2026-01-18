from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
def _BuildFileImportObject(full_path, name=None):
    if _IsUrl(full_path):
        return _ImportUrl(full_path, name)
    else:
        return _ImportFile(full_path, name)