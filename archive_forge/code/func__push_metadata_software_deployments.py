import itertools
import uuid
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import requests
from urllib import parse
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.engine import api
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import software_config_io as swc_io
from heat.objects import resource as resource_objects
from heat.objects import software_config as software_config_object
from heat.objects import software_deployment as software_deployment_object
from heat.rpc import api as rpc_api
@resource_objects.retry_on_conflict
def _push_metadata_software_deployments(self, cnxt, server_id, stack_user_project_id):
    rs = db_api.resource_get_by_physical_resource_id(cnxt, server_id)
    if not rs:
        return
    if rs.action == resource.Resource.DELETE:
        return
    deployments = self.metadata_software_deployments(cnxt, server_id)
    md = rs.rsrc_metadata or {}
    md['deployments'] = deployments
    metadata_put_url = None
    metadata_queue_id = None
    for rd in rs.data:
        if rd.key == 'metadata_put_url':
            metadata_put_url = rd.value
        if rd.key == 'metadata_queue_id':
            metadata_queue_id = rd.value
    action = _('deployments of server %s') % server_id
    atomic_key = rs.atomic_key
    rows_updated = db_api.resource_update(cnxt, rs.id, {'rsrc_metadata': md}, atomic_key)
    if not rows_updated:
        LOG.debug('Retrying server %s deployment metadata update', server_id)
        raise exception.ConcurrentTransaction(action=action)
    LOG.debug('Updated server %s deployment metadata', server_id)
    if metadata_put_url:
        json_md = jsonutils.dumps(md)
        resp = requests.put(metadata_put_url, json_md)
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            LOG.error('Failed to deliver deployment data to server %s: %s', server_id, exc)
    if metadata_queue_id:
        project = stack_user_project_id
        queue = self._get_zaqar_queue(cnxt, rs, project, metadata_queue_id)
        zaqar_plugin = cnxt.clients.client_plugin('zaqar')
        queue.post({'body': md, 'ttl': zaqar_plugin.DEFAULT_TTL})
    if metadata_put_url is not None:
        rows_updated = db_api.resource_update(cnxt, rs.id, {}, atomic_key + 1)
        if not rows_updated:
            LOG.debug('Concurrent update to server %s deployments data detected - retrying.', server_id)
            raise exception.ConcurrentTransaction(action=action)