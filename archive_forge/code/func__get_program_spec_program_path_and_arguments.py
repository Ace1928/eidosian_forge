from __future__ import absolute_import, division, print_function
import re
from os.path import exists, getsize
from socket import gaierror
from ssl import SSLError
from time import sleep
import traceback
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
from ansible.module_utils.basic import missing_required_lib
def _get_program_spec_program_path_and_arguments(self, cmd):
    if self.windowsGuest:
        '\n            we need to warp the execution of powershell into a cmd /c because\n            the call otherwise fails with "Authentication or permission failure"\n            #FIXME: Fix the unecessary invocation of cmd and run the command directly\n            '
        program_path = 'cmd.exe'
        arguments = '/c %s' % cmd
    else:
        program_path = self.get_option('executable')
        arguments = re.sub('^%s\\s*' % program_path, '', cmd)
    return (program_path, arguments)