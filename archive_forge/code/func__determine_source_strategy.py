from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
def _determine_source_strategy():
    for src in ('sourceStrategy', 'dockerStrategy', 'customStrategy'):
        strategy = build_strategy.get(src)
        if strategy:
            return strategy.get('from')
    return None