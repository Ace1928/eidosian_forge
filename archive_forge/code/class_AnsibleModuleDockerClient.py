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
class AnsibleModuleDockerClient(AnsibleDockerClientBase):

    def __init__(self, argument_spec=None, supports_check_mode=False, mutually_exclusive=None, required_together=None, required_if=None, required_one_of=None, required_by=None, min_docker_api_version=None, fail_results=None):
        self.fail_results = fail_results or {}
        merged_arg_spec = dict()
        merged_arg_spec.update(DOCKER_COMMON_ARGS)
        if argument_spec:
            merged_arg_spec.update(argument_spec)
            self.arg_spec = merged_arg_spec
        mutually_exclusive_params = []
        mutually_exclusive_params += DOCKER_MUTUALLY_EXCLUSIVE
        if mutually_exclusive:
            mutually_exclusive_params += mutually_exclusive
        required_together_params = []
        required_together_params += DOCKER_REQUIRED_TOGETHER
        if required_together:
            required_together_params += required_together
        self.module = AnsibleModule(argument_spec=merged_arg_spec, supports_check_mode=supports_check_mode, mutually_exclusive=mutually_exclusive_params, required_together=required_together_params, required_if=required_if, required_one_of=required_one_of, required_by=required_by or {})
        self.debug = False
        self.check_mode = self.module.check_mode
        self.diff = self.module._diff
        common_args = dict(((k, self.module.params[k]) for k in DOCKER_COMMON_ARGS))
        super(AnsibleModuleDockerClient, self).__init__(common_args, min_docker_api_version=min_docker_api_version)

    def call_cli(self, *args, **kwargs):
        check_rc = kwargs.pop('check_rc', False)
        data = kwargs.pop('data', None)
        cwd = kwargs.pop('cwd', None)
        environ_update = kwargs.pop('environ_update', None)
        if kwargs:
            raise TypeError("call_cli() got an unexpected keyword argument '%s'" % list(kwargs)[0])
        environment = self._environment.copy()
        if environ_update:
            environment.update(environ_update)
        rc, stdout, stderr = self.module.run_command(self._compose_cmd(args), binary_data=True, check_rc=check_rc, cwd=cwd, data=data, encoding=None, environ_update=environment, expand_user_and_vars=False, ignore_invalid_cwd=False)
        return (rc, stdout, stderr)

    def fail(self, msg, **kwargs):
        self.fail_results.update(kwargs)
        self.module.fail_json(msg=msg, **sanitize_result(self.fail_results))

    def warn(self, msg):
        self.module.warn(msg)

    def deprecate(self, msg, version=None, date=None, collection_name=None):
        self.module.deprecate(msg, version=version, date=date, collection_name=collection_name)