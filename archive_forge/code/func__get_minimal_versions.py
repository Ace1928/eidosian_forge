from __future__ import (absolute_import, division, print_function)
import abc
import os
import platform
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common._collections_compat import Mapping, Sequence
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE, BOOLEANS_FALSE
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible_collections.community.docker.plugins.module_utils.util import (  # noqa: F401, pylint: disable=unused-import
def _get_minimal_versions(self, option_minimal_versions, ignore_params=None):
    self.option_minimal_versions = dict()
    for option in self.module.argument_spec:
        if ignore_params is not None:
            if option in ignore_params:
                continue
        self.option_minimal_versions[option] = dict()
    self.option_minimal_versions.update(option_minimal_versions)
    for option, data in self.option_minimal_versions.items():
        support_docker_py = True
        support_docker_api = True
        if 'docker_py_version' in data:
            support_docker_py = self.docker_py_version >= LooseVersion(data['docker_py_version'])
        if 'docker_api_version' in data:
            support_docker_api = self.docker_api_version >= LooseVersion(data['docker_api_version'])
        data['supported'] = support_docker_py and support_docker_api
        if not data['supported']:
            if 'detect_usage' in data:
                used = data['detect_usage'](self)
            else:
                used = self.module.params.get(option) is not None
                if used and 'default' in self.module.argument_spec[option]:
                    used = self.module.params[option] != self.module.argument_spec[option]['default']
            if used:
                if 'usage_msg' in data:
                    usg = data['usage_msg']
                else:
                    usg = 'set %s option' % (option,)
                if not support_docker_api:
                    msg = 'Docker API version is %s. Minimum version required is %s to %s.'
                    msg = msg % (self.docker_api_version_str, data['docker_api_version'], usg)
                elif not support_docker_py:
                    msg = "Docker SDK for Python version is %s (%s's Python %s). Minimum version required is %s to %s. "
                    if LooseVersion(data['docker_py_version']) < LooseVersion('2.0.0'):
                        msg += DOCKERPYUPGRADE_RECOMMEND_DOCKER
                    elif self.docker_py_version < LooseVersion('2.0.0'):
                        msg += DOCKERPYUPGRADE_SWITCH_TO_DOCKER
                    else:
                        msg += DOCKERPYUPGRADE_UPGRADE_DOCKER
                    msg = msg % (docker_version, platform.node(), sys.executable, data['docker_py_version'], usg)
                else:
                    msg = 'Cannot %s with your configuration.' % (usg,)
                self.fail(msg)