from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
class Developers(base.Group):
    """Manage Apigee developers."""
    detailed_help = {'DESCRIPTION': '\n          {description}\n\n          `{command}` manages developers that want to use APIs exposed via\n          Apigee in their applications.\n          ', 'EXAMPLES': '\n          To list the email addresses of all the developers in the active Cloud\n          Platform project, run:\n\n              $ {command} list\n\n          To get that list as a JSON array and only include developers with\n          ``example.com\'\' addresses, run:\n\n              $ {command} list --format=json --filter="email:(@example.com)"\n\n          To get details about a specific developer in the active Cloud Platform\n          project, run:\n\n              $ {command} describe DEVELOPER_EMAIL\n      '}