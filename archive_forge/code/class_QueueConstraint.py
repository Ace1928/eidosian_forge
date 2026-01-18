from oslo_log import log as logging
from zaqarclient.queues.v2 import client as zaqarclient
from zaqarclient.transport import errors as zaqar_errors
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine import constraints
class QueueConstraint(constraints.BaseCustomConstraint):
    resource_client_name = CLIENT_NAME
    resource_getter_name = 'get_queue'