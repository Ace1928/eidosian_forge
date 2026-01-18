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
class FernetSetup(BasePermissionsSetup):
    """Setup key repositories for Fernet tokens and auth receipts.

    This also creates a primary key used for both creating and validating
    Fernet tokens and auth receipts. To improve security, you should rotate
    your keys (using keystone-manage fernet_rotate, for example).

    """
    name = 'fernet_setup'

    @classmethod
    def main(cls):
        keystone_user_id, keystone_group_id = cls.get_user_group()
        cls.initialize_fernet_repository(keystone_user_id, keystone_group_id, 'fernet_tokens')
        if os.path.abspath(CONF.fernet_tokens.key_repository) != os.path.abspath(CONF.fernet_receipts.key_repository):
            cls.initialize_fernet_repository(keystone_user_id, keystone_group_id, 'fernet_receipts')
        elif CONF.fernet_tokens.max_active_keys != CONF.fernet_receipts.max_active_keys:
            LOG.warning('Receipt and Token fernet key directories are the same but `max_active_keys` is different. Receipt `max_active_keys` will be ignored in favor of Token `max_active_keys`.')