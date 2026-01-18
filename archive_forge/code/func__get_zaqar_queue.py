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
def _get_zaqar_queue(self, cnxt, rs, project, queue_name):
    user = password = signed_url_data = None
    for rd in rs.data:
        if rd.key == 'password':
            password = crypt.decrypt(rd.decrypt_method, rd.value)
        if rd.key == 'user_id':
            user = rd.value
        if rd.key == 'zaqar_queue_signed_url_data':
            signed_url_data = jsonutils.loads(rd.value)
    zaqar_plugin = cnxt.clients.client_plugin('zaqar')
    if signed_url_data is None:
        keystone = cnxt.clients.client('keystone')
        token = keystone.stack_domain_user_token(user_id=user, project_id=project, password=password)
        zaqar = zaqar_plugin.create_for_tenant(project, token)
    else:
        signed_url_data.pop('project')
        zaqar = zaqar_plugin.create_from_signed_url(project, **signed_url_data)
    return zaqar.queue(queue_name)