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
def add_alembic_subparser(sub, cmd):
    return sub.add_parser(cmd, help=getattr(alembic_command, cmd).__doc__)