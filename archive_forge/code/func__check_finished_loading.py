import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
def _check_finished_loading(self, templates):
    if all((template['status'] == 'ERROR' for template in templates)):
        return True
    try:
        api_templates = utils.get_client(self).template.list()
        self._update_templates_status(api_templates, templates)
        if any((template['status'] == 'LOADING' for template in templates)):
            return False
        return True
    except Exception:
        return True