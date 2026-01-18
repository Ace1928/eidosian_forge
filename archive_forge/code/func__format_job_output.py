import sys
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import jobs as jobs_v1
def _format_job_output(app, data):
    data['status'] = data['info']['status']
    del data['info']