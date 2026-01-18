import os
import sys
import time
from alembic import command as alembic_command
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_log import log as logging
from oslo_utils import encodeutils
from glance.common import config
from glance.common import exception
from glance import context
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata
from glance.i18n import _
class DbLegacyCommands(object):
    """Class for managing the db using legacy commands"""

    def __init__(self, command_object):
        self.command_object = command_object

    def version(self):
        self.command_object.version()

    def upgrade(self, version='heads'):
        self.command_object.upgrade(CONF.command.version)

    def version_control(self, version=db_migration.ALEMBIC_INIT_VERSION):
        self.command_object.version_control(CONF.command.version)

    def sync(self, version=None):
        self.command_object.sync(CONF.command.version)

    def expand(self):
        self.command_object.expand()

    def contract(self):
        self.command_object.contract()

    def migrate(self):
        self.command_object.migrate()

    def check(self):
        self.command_object.check()

    def load_metadefs(self, path=None, merge=False, prefer_new=False, overwrite=False):
        self.command_object.load_metadefs(CONF.command.path, CONF.command.merge, CONF.command.prefer_new, CONF.command.overwrite)

    def unload_metadefs(self):
        self.command_object.unload_metadefs()

    def export_metadefs(self, path=None):
        self.command_object.export_metadefs(CONF.command.path)