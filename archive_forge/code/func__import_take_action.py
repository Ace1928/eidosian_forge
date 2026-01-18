import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def _import_take_action(self, client, parsed_args):
    if not parsed_args.image_id or not parsed_args.flavor_id:
        raise exceptions.CommandError('At least --image_id and --flavor_id should be specified')
    blob = osc_utils.read_blob_file_contents(parsed_args.json)
    try:
        template = json.loads(blob)
    except ValueError as e:
        raise exceptions.CommandError('An error occurred when reading template from file %s: %s' % (parsed_args.json, e))
    template['node_group_template']['floating_ip_pool'] = parsed_args.floating_ip_pool
    template['node_group_template']['image_id'] = parsed_args.image_id
    template['node_group_template']['flavor_id'] = parsed_args.flavor_id
    template['node_group_template']['security_groups'] = parsed_args.security_groups
    if parsed_args.name:
        template['node_group_template']['name'] = parsed_args.name
    data = client.node_group_templates.create(**template['node_group_template']).to_dict()
    return data