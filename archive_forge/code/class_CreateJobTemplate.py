from oslo_log import log as logging
from saharaclient.osc.v1 import job_templates as jt_v1
class CreateJobTemplate(jt_v1.CreateJobTemplate):
    """Creates job template"""
    log = logging.getLogger(__name__ + '.CreateJobTemplate')