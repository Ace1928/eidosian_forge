import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _add_identity_and_resource_options_to_parser(parser):
    system_or_domain_or_project = parser.add_mutually_exclusive_group()
    system_or_domain_or_project.add_argument('--system', metavar='<system>', help=_('Include <system> (all)'))
    system_or_domain_or_project.add_argument('--domain', metavar='<domain>', help=_('Include <domain> (name or ID)'))
    system_or_domain_or_project.add_argument('--project', metavar='<project>', help=_('Include <project> (name or ID)'))
    user_or_group = parser.add_mutually_exclusive_group()
    user_or_group.add_argument('--user', metavar='<user>', help=_('Include <user> (name or ID)'))
    user_or_group.add_argument('--group', metavar='<group>', help=_('Include <group> (name or ID)'))
    common.add_group_domain_option_to_parser(parser)
    common.add_project_domain_option_to_parser(parser)
    common.add_user_domain_option_to_parser(parser)
    common.add_inherited_option_to_parser(parser)