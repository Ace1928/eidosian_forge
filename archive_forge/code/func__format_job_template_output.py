import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils
from saharaclient.osc import utils
def _format_job_template_output(data):
    data['mains'] = osc_utils.format_list(['%s:%s' % (m['name'], m['id']) for m in data['mains']])
    data['libs'] = osc_utils.format_list(['%s:%s' % (l['name'], l['id']) for l in data['libs']])