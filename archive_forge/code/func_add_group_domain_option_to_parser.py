from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def add_group_domain_option_to_parser(parser):
    parser.add_argument('--group-domain', metavar='<group-domain>', help=_('Domain the group belongs to (name or ID). This can be used in case collisions between group names exist.'))