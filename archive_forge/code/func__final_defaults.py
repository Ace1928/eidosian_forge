import getpass
import logging
import sys
import traceback
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from oslo_utils import importutils
from oslo_utils import strutils
from osc_lib.cli import client_config as cloud_config
from osc_lib import clientmanager
from osc_lib.command import timing
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import logs
from osc_lib import utils
from osc_lib import version
def _final_defaults(self):
    self._auth_type = None
    project_id = getattr(self.options, 'project_id', None)
    project_name = getattr(self.options, 'project_name', None)
    tenant_id = getattr(self.options, 'tenant_id', None)
    tenant_name = getattr(self.options, 'tenant_name', None)
    if project_id and (not tenant_id):
        self.options.tenant_id = project_id
    if project_name and (not tenant_name):
        self.options.tenant_name = project_name
    if tenant_id and (not project_id):
        self.options.project_id = tenant_id
    if tenant_name and (not project_name):
        self.options.project_name = tenant_name
    self.default_domain = self.options.default_domain