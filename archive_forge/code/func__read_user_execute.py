from __future__ import absolute_import, division, print_function
import os
import platform
import pwd
import re
import sys
import tempfile
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.six.moves import shlex_quote
def _read_user_execute(self):
    """
        Returns the command line for reading a crontab
        """
    user = ''
    if self.user:
        if platform.system() == 'SunOS':
            return "su %s -c '%s -l'" % (shlex_quote(self.user), shlex_quote(self.cron_cmd))
        elif platform.system() == 'AIX':
            return '%s -l %s' % (shlex_quote(self.cron_cmd), shlex_quote(self.user))
        elif platform.system() == 'HP-UX':
            return '%s %s %s' % (self.cron_cmd, '-l', shlex_quote(self.user))
        elif pwd.getpwuid(os.getuid())[0] != self.user:
            user = '-u %s' % shlex_quote(self.user)
    return '%s %s %s' % (self.cron_cmd, user, '-l')