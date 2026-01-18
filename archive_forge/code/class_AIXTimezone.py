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
class AIXTimezone(Timezone):
    """This is a Timezone manipulation class for AIX instances.

    It uses the C(chtz) utility to set the timezone, and
    inspects C(/etc/environment) to determine the current timezone.

    While AIX time zones can be set using two formats (POSIX and
    Olson) the preferred method is Olson.
    See the following article for more information:
    https://developer.ibm.com/articles/au-aix-posix/

    NB: AIX needs to be rebooted in order for the change to be
    activated.
    """

    def __init__(self, module):
        super(AIXTimezone, self).__init__(module)
        self.settimezone = self.module.get_bin_path('chtz', required=True)

    def __get_timezone(self):
        """ Return the current value of TZ= in /etc/environment """
        try:
            f = open('/etc/environment', 'r')
            etcenvironment = f.read()
            f.close()
        except Exception:
            self.module.fail_json(msg='Issue reading contents of /etc/environment')
        match = re.search('^TZ=(.*)$', etcenvironment, re.MULTILINE)
        if match:
            return match.group(1)
        else:
            return None

    def get(self, key, phase):
        """Lookup the current timezone name in `/etc/environment`. If anything else
        is requested, or if the TZ field is not set we fail.
        """
        if key == 'name':
            return self.__get_timezone()
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)

    def set(self, key, value):
        """Set the requested timezone through chtz, an invalid timezone name
        will be rejected and we have no further input validation to perform.
        """
        if key == 'name':
            zonefile = '/usr/share/lib/zoneinfo/' + value
            try:
                if not os.path.isfile(zonefile):
                    self.module.fail_json(msg='%s is not a recognized timezone.' % value)
            except Exception:
                self.module.fail_json(msg='Failed to check %s.' % zonefile)
            cmd = 'chtz %s' % value
            rc, stdout, stderr = self.module.run_command(cmd)
            if rc != 0:
                self.module.fail_json(msg=stderr)
            TZ = self.__get_timezone()
            if TZ != value:
                msg = 'TZ value does not match post-change (Actual: %s, Expected: %s).' % (TZ, value)
                self.module.fail_json(msg=msg)
        else:
            self.module.fail_json(msg='%s is not a supported option on target platform' % key)