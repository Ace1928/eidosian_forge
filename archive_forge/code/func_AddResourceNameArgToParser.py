from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddResourceNameArgToParser(parser):
    """Adds resource name argument for the namespaced name or resource name to the parser.

  Args:
    parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('RESOURCE_NAME', metavar='RESOURCE_NAME', help='Resource name or namespaced name. The resource name should be in the form {resource_type}/{numeric_id}. The namespaced name should be in the form {org_id}/{short_name} where short_name must be 1-63 characters, beginning and ending with an alphanumeric character ([a-z0-9A-Z]) with dashes (-), underscores ( _ ), dots (.), and alphanumerics between.')