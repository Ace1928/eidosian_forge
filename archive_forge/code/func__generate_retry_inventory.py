from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible import context
from ansible.executor.task_queue_manager import TaskQueueManager, AnsibleEndPlay
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.loader import become_loader, connection_loader, shell_loader
from ansible.playbook import Playbook
from ansible.template import Templar
from ansible.utils.helpers import pct_to_int
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.utils.path import makedirs_safe
from ansible.utils.ssh_functions import set_default_transport
from ansible.utils.display import Display
def _generate_retry_inventory(self, retry_path, replay_hosts):
    """
        Called when a playbook run fails. It generates an inventory which allows
        re-running on ONLY the failed hosts.  This may duplicate some variable
        information in group_vars/host_vars but that is ok, and expected.
        """
    try:
        makedirs_safe(os.path.dirname(retry_path))
        with open(retry_path, 'w') as fd:
            for x in replay_hosts:
                fd.write('%s\n' % x)
    except Exception as e:
        display.warning("Could not create retry file '%s'.\n\t%s" % (retry_path, to_text(e)))
        return False
    return True