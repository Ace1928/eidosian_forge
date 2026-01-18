from __future__ import absolute_import, division, print_function
import os
import re
import time
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.parse import urlsplit
from ansible.plugins.action.normal import ActionModule as _ActionModule
from ansible.utils.display import Display
from ansible.utils.hashing import checksum, checksum_s
def _handle_backup_option(self, result, task_vars, backup_options):
    filename = None
    backup_path = None
    try:
        non_config_regexes = self._connection.cliconf.get_option('non_config_lines', task_vars)
    except (AttributeError, KeyError):
        non_config_regexes = []
    try:
        content = self._sanitize_contents(contents=result.pop('__backup__'), filters=non_config_regexes)
    except KeyError:
        raise AnsibleError('Failed while reading configuration backup')
    if backup_options:
        filename = backup_options.get('filename')
        backup_path = backup_options.get('dir_path')
    tstamp = time.strftime('%Y-%m-%d@%H:%M:%S', time.localtime(time.time()))
    if not backup_path:
        cwd = self._get_working_path()
        backup_path = os.path.join(cwd, 'backup')
    if not filename:
        filename = '%s_config.%s' % (task_vars['inventory_hostname'], tstamp)
    dest = os.path.join(backup_path, filename)
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    changed = False
    if not os.path.exists(dest) or checksum(dest) != checksum_s(content):
        try:
            with open(dest, 'w') as output_file:
                output_file.write(content)
        except Exception as exc:
            result['failed'] = True
            result['msg'] = 'Could not write to destination file %s: %s' % (dest, to_text(exc))
            return
        changed = True
    result['backup_path'] = dest
    result['changed'] = changed
    result['date'], result['time'] = tstamp.split('@')
    if not (backup_options and backup_options.get('filename')):
        result['filename'] = os.path.basename(result['backup_path'])
        result['shortname'] = os.path.splitext(result['backup_path'])[0]