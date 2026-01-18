from __future__ import absolute_import, division, print_function
import json  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
class PodmanVolumeModuleParams:
    """Creates list of arguments for podman CLI command.

       Arguments:
           action {str} -- action type from 'create', 'delete'
           params {dict} -- dictionary of module parameters

       """

    def __init__(self, action, params, podman_version, module):
        self.params = params
        self.action = action
        self.podman_version = podman_version
        self.module = module

    def construct_command_from_params(self):
        """Create a podman command from given module parameters.

        Returns:
           list -- list of byte strings for Popen command
        """
        if self.action in ['delete']:
            return self._simple_action()
        if self.action in ['create']:
            return self._create_action()

    def _simple_action(self):
        if self.action == 'delete':
            cmd = ['rm', '-f', self.params['name']]
            return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]

    def _create_action(self):
        cmd = [self.action, self.params['name']]
        all_param_methods = [func for func in dir(self) if callable(getattr(self, func)) and func.startswith('addparam')]
        params_set = (i for i in self.params if self.params[i] is not None)
        for param in params_set:
            func_name = '_'.join(['addparam', param])
            if func_name in all_param_methods:
                cmd = getattr(self, func_name)(cmd)
        return [to_bytes(i, errors='surrogate_or_strict') for i in cmd]

    def check_version(self, param, minv=None, maxv=None):
        if minv and LooseVersion(minv) > LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported from podman version %s only! Current version is %s' % (param, minv, self.podman_version))
        if maxv and LooseVersion(maxv) < LooseVersion(self.podman_version):
            self.module.fail_json(msg='Parameter %s is supported till podman version %s only! Current version is %s' % (param, minv, self.podman_version))

    def addparam_label(self, c):
        for label in self.params['label'].items():
            c += ['--label', b'='.join([to_bytes(l, errors='surrogate_or_strict') for l in label])]
        return c

    def addparam_driver(self, c):
        return c + ['--driver', self.params['driver']]

    def addparam_options(self, c):
        for opt in self.params['options']:
            c += ['--opt', opt]
        return c