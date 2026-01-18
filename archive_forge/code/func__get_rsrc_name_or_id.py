from ironicclient.common.apiclient import exceptions as ic_exc
from ironicclient.v1 import client as ironic_client
from oslo_config import cfg
from heat.common import exception
from heat.engine.clients import client_plugin
from heat.engine import constraints
def _get_rsrc_name_or_id(self, value, entity, entity_msg):
    entity_client = getattr(self.client(), entity)
    try:
        return entity_client.get(value).uuid
    except ic_exc.NotFound:
        raise exception.EntityNotFound(entity=entity_msg, name=value)