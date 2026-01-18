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
def _delete_executions(self):
    if self.data().get(self.EXECUTIONS):
        for id in self.data().get(self.EXECUTIONS).split(','):
            with self.client_plugin().ignore_not_found:
                self.client().executions.delete(id)
        self.data_delete('executions')