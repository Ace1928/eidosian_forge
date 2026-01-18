from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kmsinventory import inventory
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
class ListKeys(base.ListCommand):
    """Lists the keys in a project."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        pass

    def Run(self, args):
        project = properties.VALUES.core.project.Get(required=True)
        return inventory.ListKeys(project, args)