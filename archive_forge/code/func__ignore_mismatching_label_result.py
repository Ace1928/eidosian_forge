from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.text.formatters import human_to_bytes
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._platform import (
from ansible_collections.community.docker.plugins.module_utils.module_container.base import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.errors import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import (
def _ignore_mismatching_label_result(module, client, api_version, option, image, container_value, expected_value):
    if option.comparison == 'strict' and module.params['image_label_mismatch'] == 'fail':
        image_labels = _get_image_labels(image)
        would_remove_labels = []
        labels_param = module.params['labels'] or {}
        for label in image_labels:
            if label not in labels_param:
                would_remove_labels.append('"%s"' % (label,))
        if would_remove_labels:
            msg = "Some labels should be removed but are present in the base image. You can set image_label_mismatch to 'ignore' to ignore this error. Labels: {0}"
            client.fail(msg.format(', '.join(would_remove_labels)))
    return False