import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _get_nova_metadata(self, properties):
    if properties is None or properties.get(self.TAGS) is None:
        return None
    return dict(((tm[self.TAG_KEY], tm[self.TAG_VALUE]) for tm in properties[self.TAGS]))