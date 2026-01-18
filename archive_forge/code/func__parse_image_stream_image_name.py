from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def _parse_image_stream_image_name(name):
    v = name.split('@')
    if len(v) != 2:
        return (None, None, 'expected exactly one @ in the isimage name %s' % name)
    name = v[0]
    tag = v[1]
    if len(name) == 0 or len(tag) == 0:
        return (None, None, 'image stream image name %s must have a name and ID' % name)
    return (name, tag, None)