import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def _map_attrs(args, source_attr_map):
    res = {}
    for k, v in args.items():
        if v is None or k not in source_attr_map:
            continue
        source_val = source_attr_map[k]
        if len(source_val) == 2:
            res[source_val[0]] = source_val[1](v)
        elif len(source_val) == 3:
            if not isinstance(v, list):
                res[source_val[0]] = get_resource_id(source_val[2], source_val[1], v)
            else:
                res[source_val[0]] = [get_resource_id(source_val[2], source_val[1], x) for x in v]
        elif len(source_val) == 4:
            parent = source_attr_map[source_val[2]]
            parent_id = get_resource_id(parent[2], parent[1], args[source_val[2]])
            child = source_val
            res[child[0]] = get_resource_id(child[3], child[1], {child[0]: str(v), parent[0]: str(parent_id)})
    return res