from collections import abc
import datetime
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import importutils
from glance.common import exception
from glance.common import timeutils
from glance.i18n import _, _LE, _LI, _LW
def _import_delayed_delete():
    global _delayed_delete_imported
    if not _delayed_delete_imported:
        CONF.import_opt('delayed_delete', 'glance_store')
        _delayed_delete_imported = True