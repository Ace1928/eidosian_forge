import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
class UserSetup(BaseApp):
    """Create user with specified UUID."""
    name = 'user_setup'

    def __init__(self):
        self.identity = idutils.Identity()

    @classmethod
    def add_argument_parser(cls, subparsers):
        parser = super(UserSetup, cls).add_argument_parser(subparsers)
        parser.add_argument('--username', default=None, required=True, help='The username of the keystone user that is being created.')
        parser.add_argument('--user-password-plain', default=None, required=True, help='The plaintext password for the keystone user that is being created.')
        parser.add_argument('--user-id', default=None, help='The UUID of the keystone user being created.')
        return parser

    def do_user_setup(self):
        """Create user with specified UUID."""
        self.identity.user_name = CONF.command.username
        self.identity.user_password = CONF.command.user_password_plain
        self.identity.user_id = CONF.command.user_id
        self.identity.user_setup()

    @classmethod
    def main(cls):
        klass = cls()
        klass.do_user_setup()