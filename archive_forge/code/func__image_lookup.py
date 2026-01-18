from __future__ import (absolute_import, division, print_function)
import abc
import json
import shlex
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils._api.auth import resolve_repository_name
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def _image_lookup(self, name, tag):
    """
        Including a tag in the name parameter sent to the Docker SDK for Python images method
        does not work consistently. Instead, get the result set for name and manually check
        if the tag exists.
        """
    dummy, images, dummy = self.call_cli_json_stream('image', 'ls', '--format', '{{ json . }}', '--no-trunc', '--filter', 'reference={0}'.format(name), check_rc=True)
    if tag:
        lookup = '%s:%s' % (name, tag)
        lookup_digest = '%s@%s' % (name, tag)
        response = images
        images = []
        for image in response:
            if image.get('Tag') == tag or image.get('Digest') == tag:
                images = [image]
                break
    return images