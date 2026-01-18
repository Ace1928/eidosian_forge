from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _update_homebrew(self):
    rc, out, err = self.module.run_command([self.brew_path, 'update'])
    if rc == 0:
        if out and isinstance(out, string_types):
            already_updated = any((re.search('Already up-to-date.', s.strip(), re.IGNORECASE) for s in out.split('\n') if s))
            if not already_updated:
                self.changed = True
                self.message = 'Homebrew updated successfully.'
            else:
                self.message = 'Homebrew already up-to-date.'
        return True
    else:
        self.failed = True
        self.message = err.strip()
        raise HomebrewCaskException(self.message)