from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _workspace_cmd(bin_path, project_path, action, workspace):
    command = [bin_path, 'workspace', action, workspace, '-no-color']
    rc, out, err = module.run_command(command, check_rc=True, cwd=project_path)
    return (rc, out, err)