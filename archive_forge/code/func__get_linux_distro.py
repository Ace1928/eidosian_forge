from __future__ import (absolute_import, division, print_function)
import bisect
import json
import pkgutil
import re
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.distro import LinuxDistribution
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_versioned_doclink
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.facts.system.distribution import Distribution
from traceback import format_exc
def _get_linux_distro(platform_info):
    dist_result = platform_info.get('platform_dist_result', [])
    if len(dist_result) == 3 and any(dist_result):
        return (dist_result[0], dist_result[1])
    osrelease_content = platform_info.get('osrelease_content')
    if not osrelease_content:
        return (u'', u'')
    osr = LinuxDistribution._parse_os_release_content(osrelease_content)
    return (osr.get('id', u''), osr.get('version_id', u''))