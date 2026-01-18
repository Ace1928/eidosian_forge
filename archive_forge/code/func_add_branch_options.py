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
def add_branch_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--expand', action='store_true')
    group.add_argument('--contract', action='store_true')
    return group