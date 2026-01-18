from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import folders
from googlecloudsdk.calliope import base
class Folders(base.Group):
    """Manage Cloud Folders.

  Commands to query and update your Cloud Folders.
  """

    @staticmethod
    def Args(parser):
        parser.display_info.AddUriFunc(folders.GetUri)