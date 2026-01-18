from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types, string_types
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
import traceback
def _determine_list_all_kwargs(version):
    gitlab_version = LooseVersion(version)
    if gitlab_version >= LooseVersion('4.0.0'):
        return {'iterator': True, 'per_page': 100}
    elif gitlab_version >= LooseVersion('3.7.0'):
        return {'as_list': False, 'get_all': True, 'per_page': 100}
    else:
        return {'as_list': False, 'all': True, 'per_page': 100}