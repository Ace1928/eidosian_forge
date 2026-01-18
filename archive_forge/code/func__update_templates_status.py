import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
@staticmethod
def _update_templates_status(api_templates, templates):
    for template in templates:
        uuid = template.get('uuid')
        if uuid:
            api_template = find_template_with_uuid(uuid, api_templates)
            if api_template:
                template['status'] = api_template['status']