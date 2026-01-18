import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
class AnsibleTurboModule(ansible.module_utils.basic.AnsibleModule):
    embedded_in_server = False
    collection_name = None

    def __init__(self, *args, **kwargs):
        self.embedded_in_server = sys.argv[0].endswith('/server.py')
        self.collection_name = AnsibleTurboModule.collection_name or get_collection_name_from_path()
        ansible.module_utils.basic.AnsibleModule.__init__(self, *args, bypass_checks=not self.embedded_in_server, **kwargs)
        self._running = None
        if not self.embedded_in_server:
            self.run_on_daemon()

    def socket_path(self):
        if self._remote_tmp is None:
            abs_remote_tmp = tempfile.gettempdir()
        else:
            abs_remote_tmp = os.path.expanduser(os.path.expandvars(self._remote_tmp))
        return os.path.join(abs_remote_tmp, f'turbo_mode.{self.collection_name}.socket')

    def init_args(self):
        argument_specs = expand_argument_specs_aliases(self.argument_spec)
        args = prepare_args(argument_specs, self.params)
        for k in ansible.module_utils.basic.PASS_VARS:
            attribute = ansible.module_utils.basic.PASS_VARS[k][0]
            if not hasattr(self, attribute):
                continue
            v = getattr(self, attribute)
            if isinstance(v, int) or isinstance(v, bool) or isinstance(v, str):
                args['ANSIBLE_MODULE_ARGS'][f'_ansible_{k}'] = v
        return args

    def run_on_daemon(self):
        result = dict(changed=False, original_message='', message='')
        ttl = os.environ.get('ANSIBLE_TURBO_LOOKUP_TTL', None)
        with ansible_collections.cloud.common.plugins.module_utils.turbo.common.connect(socket_path=self.socket_path(), ttl=ttl) as turbo_socket:
            ansiblez_path = sys.path[0]
            args = self.init_args()
            data = [ansiblez_path, json.dumps(args), dict(os.environ)]
            content = json.dumps(data).encode()
            result = turbo_socket.communicate(content)
        self.exit_json(**result)

    def exit_json(self, **kwargs):
        if not self.embedded_in_server:
            super().exit_json(**kwargs)
        else:
            self.do_cleanup_files()
            raise EmbeddedModuleSuccess(**kwargs)

    def fail_json(self, *args, **kwargs):
        if not self.embedded_in_server:
            super().fail_json(**kwargs)
        else:
            self.do_cleanup_files()
            raise EmbeddedModuleFailure(*args, **kwargs)