import collections
import email
from email.mime import multipart
from email.mime import text
import os
import pkgutil
import string
from urllib import parse as urlparse
from neutronclient.common import exceptions as q_exceptions
from novaclient import api_versions
from novaclient import client as nc
from novaclient import exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_exception
from heat.engine.clients import client_plugin
from heat.engine.clients import microversion_mixin
from heat.engine.clients import os as os_client
from heat.engine import constraints
@tenacity.retry(stop=tenacity.stop_after_attempt(max(cfg.CONF.client_retry_limit + 1, 0)), retry=tenacity.retry_if_exception(client_plugin.retry_if_connection_err), reraise=True)
def absolute_limits(self):
    """Return the absolute limits as a dictionary."""
    limits = self.client().limits.get()
    return dict([(limit.name, limit.value) for limit in list(limits.absolute)])