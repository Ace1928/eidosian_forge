from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _delete_zaqar_signal_queue(self):
    queue_id = self.data().get('zaqar_signal_queue_id')
    if not queue_id:
        return
    zaqar_plugin = self.client_plugin('zaqar')
    zaqar = zaqar_plugin.create_for_tenant(self.stack.stack_user_project_id, self._user_token())
    with zaqar_plugin.ignore_not_found:
        zaqar.queue(queue_id).delete()
    self.data_delete('zaqar_signal_queue_id')