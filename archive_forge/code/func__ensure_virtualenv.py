from __future__ import absolute_import, division, print_function
import os
import sys
import shlex
from ansible.module_utils.basic import AnsibleModule
def _ensure_virtualenv(module):
    venv_param = module.params['virtualenv']
    if venv_param is None:
        return
    vbin = os.path.join(venv_param, 'bin')
    activate = os.path.join(vbin, 'activate')
    if not os.path.exists(activate):
        if not module.params['ack_venv_creation_deprecation']:
            module.deprecate('The behavior of "creating the virtual environment when missing" is being deprecated and will be removed in community.general version 9.0.0. Set the module parameter `ack_venv_creation_deprecation: true` to prevent this message from showing up when creating a virtualenv.', version='9.0.0', collection_name='community.general')
        virtualenv = module.get_bin_path('virtualenv', True)
        vcmd = [virtualenv, venv_param]
        rc, out_venv, err_venv = module.run_command(vcmd)
        if rc != 0:
            _fail(module, vcmd, out_venv, err_venv)
    os.environ['PATH'] = '%s:%s' % (vbin, os.environ['PATH'])
    os.environ['VIRTUAL_ENV'] = venv_param