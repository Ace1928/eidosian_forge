from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def _parse_image_stream_tag_name(name):
    if '@' in name:
        return (None, None, '%s is an image stream image, not an image stream tag' % name)
    v = name.split(':')
    if len(v) != 2:
        return (None, None, 'expected exactly one : delimiter in the istag %s' % name)
    name = v[0]
    tag = v[1]
    if len(name) == 0 or len(tag) == 0:
        return (None, None, 'image stream tag name %s must have a name and a tag' % name)
    return (name, tag, None)