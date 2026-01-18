from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _container_create_clone(self):
    """Clone a new LXC container from an existing container.

        This method will clone an existing container to a new container using
        the `clone_name` variable as the new container name. The method will
        create a container if the container `name` does not exist.

        Note that cloning a container will ensure that the original container
        is "stopped" before the clone can be done. Because this operation can
        require a state change the method will return the original container
        to its prior state upon completion of the clone.

        Once the clone is complete the new container will be left in a stopped
        state.
        """
    container_state = self._get_state()
    if container_state != 'stopped':
        self.state_change = True
        self.container.stop()
    clone_vars = 'variables-lxc-copy'
    clone_cmd = self.module.get_bin_path('lxc-copy')
    if not clone_cmd:
        clone_vars = 'variables-lxc-clone'
        clone_cmd = self.module.get_bin_path('lxc-clone', True)
    build_command = [clone_cmd]
    build_command = self._add_variables(variables_dict=self._get_vars(variables=LXC_COMMAND_MAP['clone'][clone_vars]), build_command=build_command)
    if self.module.params['clone_snapshot']:
        build_command.append('--snapshot')
    elif self.module.params['backing_store'] == 'overlayfs':
        build_command.append('--snapshot')
    rc, return_data, err = self.module.run_command(build_command)
    if rc != 0:
        message = 'Failed executing %s.' % os.path.basename(clone_cmd)
        self.failure(err=err, rc=rc, msg=message, command=' '.join(build_command))
    else:
        self.state_change = True
        if container_state == 'running':
            self.container.start()
        elif container_state == 'frozen':
            self.container.start()
            self.container.freeze()
    return True