import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
def _take_action(self, client, parsed_args):
    update_dict = utils.create_dict_from_kwargs(is_public=parsed_args.is_public, is_protected=parsed_args.is_protected)
    data = utils.update_job(client, self.app, parsed_args, update_dict)
    return data