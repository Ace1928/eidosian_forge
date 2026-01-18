from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _alternate_retrieval_method(module):
    get_argument = 'echo "%s" | gnome-keyring-daemon --unlock\nkeyring get %s %s\n' % (quote(module.params['keyring_password']), quote(module.params['service']), quote(module.params['username']))
    dummy, stdout, dummy = module.run_command('dbus-run-session -- /bin/bash', use_unsafe_shell=True, data=get_argument, encoding=None)
    try:
        return stdout.decode('UTF-8').splitlines()[1]
    except IndexError:
        return None