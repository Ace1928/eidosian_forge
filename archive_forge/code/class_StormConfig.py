from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
class StormConfig(SSHConfig):

    def parse(self, file_obj):
        """
        Read an OpenSSH config from the given file object.
        @param file_obj: a file-like object to read the config file from
        @type file_obj: file
        """
        order = 1
        host = {'host': ['*'], 'config': {}}
        for line in file_obj:
            line = line.rstrip('\n').lstrip()
            if line == '':
                self._config.append({'type': 'empty_line', 'value': line, 'host': '', 'order': order})
                order += 1
                continue
            if line.startswith('#'):
                self._config.append({'type': 'comment', 'value': line, 'host': '', 'order': order})
                order += 1
                continue
            if '=' in line:
                if line.lower().strip().startswith('proxycommand'):
                    proxy_re = re.compile('^(proxycommand)\\s*=*\\s*(.*)', re.I)
                    match = proxy_re.match(line)
                    key, value = (match.group(1).lower(), match.group(2))
                else:
                    key, value = line.split('=', 1)
                    key = key.strip().lower()
            else:
                i = 0
                while i < len(line) and (not line[i].isspace()):
                    i += 1
                if i == len(line):
                    raise Exception('Unparsable line: %r' % line)
                key = line[:i].lower()
                value = line[i:].lstrip()
            if key == 'host':
                self._config.append(host)
                value = value.split()
                host = {key: value, 'config': {}, 'type': 'entry', 'order': order}
                order += 1
            elif key in ['identityfile', 'localforward', 'remoteforward']:
                if key in host['config']:
                    host['config'][key].append(value)
                else:
                    host['config'][key] = [value]
            elif key not in host['config']:
                host['config'].update({key: value})
        self._config.append(host)