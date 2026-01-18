from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import appinfo_includes
from googlecloudsdk.third_party.appengine.api import croninfo
from googlecloudsdk.third_party.appengine.api import dispatchinfo
from googlecloudsdk.third_party.appengine.api import queueinfo
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.datastore import datastore_index
class YamlParseError(Error):
    """An exception for when a specific yaml file is not well formed."""

    def __init__(self, file_path, e):
        """Creates a new Error.

    Args:
      file_path: str, The full path of the file that failed to parse.
      e: Exception, The exception that was originally raised.
    """
        super(YamlParseError, self).__init__('An error occurred while parsing file: [{file_path}]\n{err}'.format(file_path=file_path, err=e))