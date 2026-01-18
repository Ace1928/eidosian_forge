from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
def SortByName(list_response):
    """Return the list_response sorted by name."""
    return sorted(list_response, key=lambda x: x.name)