import copy
from oslo_serialization import jsonutils
import yaml
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import signal_responder
from heat.engine import support
from heat.engine import translation
def _workflow_name(self):
    return self.properties.get(self.NAME) or self.physical_resource_name()