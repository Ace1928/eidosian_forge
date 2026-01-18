from __future__ import absolute_import, division, print_function
import_status:
from ansible.module_utils.basic import (
from ansible.module_utils.urls import (
from ..module_utils.api import (
from ansible.module_utils._text import (
def _transform_import_to_image(self, imp):
    img = imp.get('custom_image', {})
    return {'href': img.get('href'), 'uuid': imp['uuid'], 'name': img.get('name'), 'created_at': None, 'size_gb': None, 'checksums': None, 'tags': imp['tags'], 'url': imp['url'], 'import_status': imp['status'], 'error_message': imp.get('error_message', ''), 'state': 'present', 'user_data_handling': self._module.params['user_data_handling'], 'zones': self._module.params['zones'], 'slug': self._module.params['slug'], 'firmware_type': self._module.params['firmware_type']}