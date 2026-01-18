from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _get_container_image(self, container, fallback=None):
    if not container.exists or container.removing:
        return fallback
    image = container.image
    if is_image_name_id(image):
        image = self.engine_driver.inspect_image_by_id(self.client, image)
    else:
        repository, tag = parse_repository_tag(image)
        if not tag:
            tag = 'latest'
        image = self.engine_driver.inspect_image_by_name(self.client, repository, tag)
    return image or fallback