from __future__ import absolute_import, division, print_function
import errno
import os
import platform
import random
import re
import string
import filecmp
from ansible.module_utils.basic import AnsibleModule, get_distribution
from ansible.module_utils.six import iteritems
class DarwinTimezone(Timezone):
    """This is the timezone implementation for Darwin which, unlike other *BSD
    implementations, uses the `systemsetup` command on Darwin to check/set
    the timezone.
    """
    regexps = dict(name=re.compile('^\\s*Time ?Zone\\s*:\\s*([^\\s]+)', re.MULTILINE))

    def __init__(self, module):
        super(DarwinTimezone, self).__init__(module)
        self.systemsetup = module.get_bin_path('systemsetup', required=True)
        self.status = dict()
        if 'name' in self.value:
            self._verify_timezone()

    def _get_current_timezone(self, phase):
        """Lookup the current timezone via `systemsetup -gettimezone`."""
        if phase not in self.status:
            self.status[phase] = self.execute(self.systemsetup, '-gettimezone')
        return self.status[phase]

    def _verify_timezone(self):
        tz = self.value['name']['planned']
        out = self.execute(self.systemsetup, '-listtimezones').splitlines()[1:]
        tz_list = list(map(lambda x: x.strip(), out))
        if tz not in tz_list:
            self.abort('given timezone "%s" is not available' % tz)
        return tz

    def get(self, key, phase):
        if key == 'name':
            status = self._get_current_timezone(phase)
            value = self.regexps[key].search(status).group(1)
            return value
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)

    def set(self, key, value):
        if key == 'name':
            self.execute(self.systemsetup, '-settimezone', value, log=True)
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)