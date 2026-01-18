import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def _export_take_action(self, client, parsed_args):
    ngt_id = get_resource_id(client.node_group_templates, parsed_args.node_group_template)
    response = client.node_group_templates.export(ngt_id)
    result = json.dumps(response._info, indent=4) + '\n'
    if parsed_args.file:
        with open(parsed_args.file, 'w+') as file:
            file.write(result)
    else:
        sys.stdout.write(result)