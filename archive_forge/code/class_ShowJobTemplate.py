from oslo_log import log as logging
from saharaclient.osc.v1 import job_templates as jt_v1
class ShowJobTemplate(jt_v1.ShowJobTemplate):
    """Display job template details"""
    log = logging.getLogger(__name__ + '.ShowJobTemplate')