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
def _service_swift_signal(self):
    swift_client = self.client('swift')
    try:
        container = swift_client.get_container(self.stack.id)
    except Exception as exc:
        self.client_plugin('swift').ignore_not_found(exc)
        LOG.debug('Swift container %s was not found', self.stack.id)
        return
    index = container[1]
    if not index:
        LOG.debug('Swift objects in container %s were not found', self.stack.id)
        return
    object_name = self.physical_resource_name()
    filtered = [obj for obj in index if object_name in obj['name']]
    signal_names = []
    for obj in filtered:
        try:
            signal = swift_client.get_object(self.stack.id, obj['name'])
        except Exception as exc:
            self.client_plugin('swift').ignore_not_found(exc)
            continue
        body = signal[1]
        if body == swift.IN_PROGRESS:
            continue
        signal_names.append(obj['name'])
        if body == '':
            self.signal(details={})
            continue
        try:
            self.signal(details=jsonutils.loads(body))
        except ValueError:
            raise exception.Error(_('Failed to parse JSON data: %s') % body)
    for signal_name in signal_names:
        if signal_name != object_name:
            swift_client.delete_object(self.stack.id, signal_name)
    if object_name in signal_names:
        swift_client.delete_object(self.stack.id, object_name)