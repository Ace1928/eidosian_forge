from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import re
import string
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
@classmethod
def IsStorageUrl(cls, path):
    try:
        cls.FromUrl(path)
    except ValueError:
        return False
    return True