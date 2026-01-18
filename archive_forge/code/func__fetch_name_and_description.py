from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import progress
from heat.engine import resource
from heat.engine import rsrc_defn
def _fetch_name_and_description(self, name=None, description=None):
    return {'name': name or self._name(), 'description': description or self._description()}