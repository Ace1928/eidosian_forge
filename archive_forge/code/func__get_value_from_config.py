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
def _get_value_from_config(self, key, phase):
    filename = self.conf_files[key]
    try:
        file = open(filename, mode='r')
    except IOError as err:
        if self._allow_ioerror(err, key):
            if key == 'hwclock':
                return 'n/a'
            elif key == 'adjtime':
                return 'UTC'
            elif key == 'name':
                return 'n/a'
        else:
            self.abort('tried to configure %s using a file "%s", but could not read it' % (key, filename))
    else:
        status = file.read()
        file.close()
        try:
            value = self.regexps[key].search(status).group(1)
        except AttributeError:
            if key == 'hwclock':
                return 'n/a'
            elif key == 'adjtime':
                return 'UTC'
            elif key == 'name':
                if phase == 'before':
                    return 'n/a'
                else:
                    self.abort('tried to configure %s using a file "%s", but could not find a valid value in it' % (key, filename))
        else:
            if key == 'hwclock':
                if self.module.boolean(value):
                    value = 'UTC'
                else:
                    value = 'local'
            elif key == 'adjtime':
                if value != 'UTC':
                    value = value.lower()
    return value