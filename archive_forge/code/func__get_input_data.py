import eventlet.queue
import functools
from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import sync_point
from heat.objects import resource as resource_objects
from heat.rpc import api as rpc_api
from heat.rpc import listener_client
def _get_input_data(req_node, input_forward_data=None):
    if req_node.is_update:
        if input_forward_data is None:
            return rsrc.node_data().as_dict()
        else:
            return input_forward_data
    elif req_node.rsrc_id != graph_key.rsrc_id:
        return rsrc.replaced_by if rsrc.replaced_by is not None else resource_id
    return None