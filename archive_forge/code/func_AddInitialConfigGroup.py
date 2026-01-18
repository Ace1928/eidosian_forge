from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddInitialConfigGroup(parser, help_text='Repository initialization configuration.'):
    """Add flags for initial config."""
    group = parser.add_group(required=False, help=help_text)
    group.add_argument('--default-branch', dest='default_branch', required=False, help='Default branch name of the repository.')
    group.add_argument('--gitignores', dest='gitignores', metavar='GITIGNORES', type=arg_parsers.ArgList(), required=False, default=[], help='List of gitignore template names user can choose from. Full list can be found here: https://cloud.google.com/secure-source-manager/docs/reference/rest/v1/projects.locations.repositories#InitialConfig')
    group.add_argument('--license', dest='license', required=False, help='License template name user can choose from. Full list can be found here: https://cloud.google.com/secure-source-manager/docs/reference/rest/v1/projects.locations.repositories#InitialConfig')
    group.add_argument('--readme', dest='readme', required=False, help='README template name. Valid template name(s) are: default.')