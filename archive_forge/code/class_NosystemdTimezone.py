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
class NosystemdTimezone(Timezone):
    """This is a Timezone manipulation class for non systemd-powered Linux.

    For timezone setting, it edits the following file and reflect changes:
        - /etc/sysconfig/clock  ... RHEL/CentOS
        - /etc/timezone         ... Debian/Ubuntu
    For hwclock setting, it executes `hwclock --systohc` command with the
    '--utc' or '--localtime' option.
    """
    conf_files = dict(name=None, hwclock=None, adjtime='/etc/adjtime')
    allow_no_file = dict(name=True, hwclock=True, adjtime=True)
    regexps = dict(name=None, hwclock=re.compile('^UTC\\s*=\\s*([^\\s]+)', re.MULTILINE), adjtime=re.compile('^(UTC|LOCAL)$', re.MULTILINE))
    dist_regexps = dict(SuSE=re.compile('^TIMEZONE\\s*=\\s*"?([^"\\s]+)"?', re.MULTILINE), redhat=re.compile('^ZONE\\s*=\\s*"?([^"\\s]+)"?', re.MULTILINE))
    dist_tzline_format = dict(SuSE='TIMEZONE="%s"\n', redhat='ZONE="%s"\n')

    def __init__(self, module):
        super(NosystemdTimezone, self).__init__(module)
        planned_tz = ''
        if 'name' in self.value:
            tzfile = self._verify_timezone()
            planned_tz = self.value['name']['planned']
            self.update_timezone = ['%s --remove-destination %s /etc/localtime' % (self.module.get_bin_path('cp', required=True), tzfile)]
        self.update_hwclock = self.module.get_bin_path('hwclock', required=True)
        distribution = get_distribution()
        self.conf_files['name'] = '/etc/timezone'
        self.regexps['name'] = re.compile('^([^\\s]+)', re.MULTILINE)
        self.tzline_format = '%s\n'
        if self.module.get_bin_path('dpkg-reconfigure') is not None:
            if 'name' in self.value:
                self.update_timezone = ['%s -sf %s /etc/localtime' % (self.module.get_bin_path('ln', required=True), tzfile), '%s --frontend noninteractive tzdata' % self.module.get_bin_path('dpkg-reconfigure', required=True)]
            self.conf_files['hwclock'] = '/etc/default/rcS'
        elif distribution == 'Alpine' or distribution == 'Gentoo':
            self.conf_files['hwclock'] = '/etc/conf.d/hwclock'
            if distribution == 'Alpine':
                self.update_timezone = ['%s -z %s' % (self.module.get_bin_path('setup-timezone', required=True), planned_tz)]
        else:
            if self.module.get_bin_path('tzdata-update') is not None:
                if not os.path.islink('/etc/localtime'):
                    self.update_timezone = [self.module.get_bin_path('tzdata-update', required=True)]
            self.conf_files['name'] = '/etc/sysconfig/clock'
            self.conf_files['hwclock'] = '/etc/sysconfig/clock'
            try:
                f = open(self.conf_files['name'], 'r')
            except IOError as err:
                if self._allow_ioerror(err, 'name'):
                    if distribution == 'SuSE':
                        self.regexps['name'] = self.dist_regexps['SuSE']
                        self.tzline_format = self.dist_tzline_format['SuSE']
                    else:
                        self.regexps['name'] = self.dist_regexps['redhat']
                        self.tzline_format = self.dist_tzline_format['redhat']
                else:
                    self.abort('could not read configuration file "%s"' % self.conf_files['name'])
            else:
                sysconfig_clock = f.read()
                f.close()
                if re.search('^TIMEZONE\\s*=', sysconfig_clock, re.MULTILINE):
                    self.regexps['name'] = self.dist_regexps['SuSE']
                    self.tzline_format = self.dist_tzline_format['SuSE']
                else:
                    self.regexps['name'] = self.dist_regexps['redhat']
                    self.tzline_format = self.dist_tzline_format['redhat']

    def _allow_ioerror(self, err, key):
        if err.errno != errno.ENOENT:
            return False
        return self.allow_no_file.get(key, False)

    def _edit_file(self, filename, regexp, value, key):
        """Replace the first matched line with given `value`.

        If `regexp` matched more than once, other than the first line will be deleted.

        Args:
            filename: The name of the file to edit.
            regexp:   The regular expression to search with.
            value:    The line which will be inserted.
            key:      For what key the file is being edited.
        """
        try:
            file = open(filename, 'r')
        except IOError as err:
            if self._allow_ioerror(err, key):
                lines = []
            else:
                self.abort('tried to configure %s using a file "%s", but could not read it' % (key, filename))
        else:
            lines = file.readlines()
            file.close()
        matched_indices = []
        for i, line in enumerate(lines):
            if regexp.search(line):
                matched_indices.append(i)
        if len(matched_indices) > 0:
            insert_line = matched_indices[0]
        else:
            insert_line = 0
        for i in matched_indices[::-1]:
            del lines[i]
        lines.insert(insert_line, value)
        try:
            file = open(filename, 'w')
        except IOError:
            self.abort('tried to configure %s using a file "%s", but could not write to it' % (key, filename))
        else:
            file.writelines(lines)
            file.close()
        self.msg.append('Added 1 line and deleted %s line(s) on %s' % (len(matched_indices), filename))

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

    def get(self, key, phase):
        planned = self.value[key]['planned']
        if key == 'hwclock':
            value = self._get_value_from_config(key, phase)
            if value == planned:
                value = self._get_value_from_config('adjtime', phase)
        elif key == 'name':
            value = self._get_value_from_config(key, phase)
            if value == planned:
                if os.path.islink('/etc/localtime'):
                    if os.path.exists('/etc/localtime'):
                        path = os.readlink('/etc/localtime')
                        linktz = re.search('(?:/(?:usr/share|etc)/zoneinfo/)(.*)', path, re.MULTILINE)
                        if linktz:
                            valuelink = linktz.group(1)
                            if valuelink != planned:
                                value = valuelink
                        else:
                            value = 'n/a'
                    else:
                        value = 'n/a'
                else:
                    try:
                        if not filecmp.cmp('/etc/localtime', '/usr/share/zoneinfo/' + planned):
                            return 'n/a'
                    except Exception:
                        return 'n/a'
        else:
            self.abort('unknown parameter "%s"' % key)
        return value

    def set_timezone(self, value):
        self._edit_file(filename=self.conf_files['name'], regexp=self.regexps['name'], value=self.tzline_format % value, key='name')
        for cmd in self.update_timezone:
            self.execute(cmd)

    def set_hwclock(self, value):
        if value == 'local':
            option = '--localtime'
            utc = 'no'
        else:
            option = '--utc'
            utc = 'yes'
        if self.conf_files['hwclock'] is not None:
            self._edit_file(filename=self.conf_files['hwclock'], regexp=self.regexps['hwclock'], value='UTC=%s\n' % utc, key='hwclock')
        self.execute(self.update_hwclock, '--systohc', option, log=True)

    def set(self, key, value):
        if key == 'name':
            self.set_timezone(value)
        elif key == 'hwclock':
            self.set_hwclock(value)
        else:
            self.abort('unknown parameter "%s"' % key)