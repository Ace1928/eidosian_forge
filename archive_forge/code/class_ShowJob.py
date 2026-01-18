import sys
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import jobs as jobs_v1
class ShowJob(jobs_v1.ShowJob):
    """Display job details"""
    log = logging.getLogger(__name__ + '.ShowJob')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = client.jobs.get(parsed_args.job).to_dict()
        _format_job_output(self.app, data)
        data = utils.prepare_data(data, jobs_v1.JOB_FIELDS)
        return self.dict2columns(data)