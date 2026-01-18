from __future__ import (absolute_import, division, print_function)
import json  # noqa: F402
import os  # noqa: F402
import shlex  # noqa: F402
from ansible.module_utils._text import to_bytes, to_native  # noqa: F402
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import normalize_signal
from ansible_collections.containers.podman.plugins.module_utils.podman.common import ARGUMENTS_OPTS_DICT
def _createcommand(self, argument):
    """Returns list of values for given argument from CreateCommand
        from Podman container inspect output.

        Args:
            argument (str): argument name

        Returns:

            all_values: list of values for given argument from createcommand
        """
    if 'createcommand' not in self.info['config']:
        return []
    cr_com = self.info['config']['createcommand']
    argument_values = ARGUMENTS_OPTS_DICT.get(argument, [argument])
    all_values = []
    for arg in argument_values:
        for ind, cr_opt in enumerate(cr_com):
            if arg == cr_opt:
                if not cr_com[ind + 1].startswith('-'):
                    all_values.append(cr_com[ind + 1])
                else:
                    return [True]
            if cr_opt.startswith('%s=' % arg):
                all_values.append(cr_opt.split('=', 1)[1])
    return all_values