import sys
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import jobs as jobs_v1
class ListJobs(jobs_v1.ListJobs):
    """Lists jobs"""
    log = logging.getLogger(__name__ + '.ListJobs')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = client.jobs.list()
        for job in data:
            job.status = job.info['status']
        if parsed_args.status:
            data = [job for job in data if job.info['status'] == parsed_args.status.replace('-', '').upper()]
        if parsed_args.long:
            columns = ('id', 'cluster id', 'job template id', 'status', 'start time', 'end time')
            column_headers = utils.prepare_column_headers(columns)
        else:
            columns = ('id', 'cluster id', 'job template id', 'status')
            column_headers = utils.prepare_column_headers(columns)
        return (column_headers, (osc_utils.get_item_properties(s, columns) for s in data))