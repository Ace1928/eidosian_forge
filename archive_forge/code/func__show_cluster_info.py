import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
def _show_cluster_info(self, data, provision_steps, parsed_args):
    fields = []
    if parsed_args.verification:
        ver_data, fields = _prepare_health_checks(data)
        data.update(ver_data)
    fields.extend(CLUSTER_FIELDS)
    data = self.dict2columns(utils.prepare_data(data, fields))
    if parsed_args.show_progress:
        output_steps = []
        for step in provision_steps:
            st_name, st_type = (step['step_name'], step['step_type'])
            description = '%s: %s' % (st_type, st_name)
            if step['successful'] is None:
                progress = 'Step in progress'
            elif step['successful']:
                progress = 'Step completed successfully'
            else:
                progress = 'Step has failed events'
            output_steps += [(description, progress)]
        data = utils.extend_columns(data, output_steps)
    return data