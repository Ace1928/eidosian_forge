from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import connection_context
@classmethod
def SetCompleteApiEndpoint(cls, complete_api_endpoint):
    cls.complete_api_endpoint = complete_api_endpoint