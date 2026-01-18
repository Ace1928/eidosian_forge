from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Applications(base.Group):
    """Manage third-party applications which call Apigee API proxies."""
    detailed_help = {'DESCRIPTION': '\n          {description}\n\n          `{command}` manages applications that want to use APIs exposed via\n          Apigee.\n          ', 'EXAMPLES': '\n          To get the names and UUIDs of all applications in the active Cloud\n          Platform project, run:\n\n              $ {command} list\n\n          To get a JSON representation of an application in the active Cloud\n          Platform project, including its API keys, run:\n\n              $ {command} describe APP_UUID --format=json\n          '}