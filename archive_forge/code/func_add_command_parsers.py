import copy
import sys
from alembic import command as alembic_command
from alembic import script as alembic_script
from alembic import util as alembic_util
from oslo_config import cfg
from oslo_log import log
import pbr.version
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.i18n import _
def add_command_parsers(subparsers):
    for name in ['current', 'history', 'branches', 'heads']:
        parser = add_alembic_subparser(subparsers, name)
        parser.set_defaults(func=do_generic_show)
        parser.add_argument('--verbose', action='store_true', help='Display more verbose output for the specified command')
    parser = add_alembic_subparser(subparsers, 'upgrade')
    parser.add_argument('--delta', type=int)
    parser.add_argument('--sql', action='store_true')
    parser.add_argument('revision', nargs='?')
    add_branch_options(parser)
    parser.set_defaults(func=do_upgrade)
    parser = subparsers.add_parser('validate', help=alembic_command.branches.__doc__ + ' and validate head file')
    parser.set_defaults(func=do_validate)
    parser = add_alembic_subparser(subparsers, 'revision')
    parser.add_argument('-m', '--message')
    parser.add_argument('--sql', action='store_true')
    group = add_branch_options(parser)
    group.add_argument('--autogenerate', action='store_true')
    parser.set_defaults(func=do_revision)