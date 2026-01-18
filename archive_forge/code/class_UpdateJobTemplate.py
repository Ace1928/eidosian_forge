from oslo_log import log as logging
from saharaclient.osc.v1 import job_templates as jt_v1
class UpdateJobTemplate(jt_v1.UpdateJobTemplate):
    """Updates job template"""
    log = logging.getLogger(__name__ + '.UpdateJobTemplate')