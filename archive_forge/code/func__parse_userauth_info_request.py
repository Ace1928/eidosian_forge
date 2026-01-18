import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _parse_userauth_info_request(self, m):
    if self.auth_method != 'keyboard-interactive':
        raise SSHException('Illegal info request from server')
    title = m.get_text()
    instructions = m.get_text()
    m.get_binary()
    prompts = m.get_int()
    prompt_list = []
    for i in range(prompts):
        prompt_list.append((m.get_text(), m.get_boolean()))
    response_list = self.interactive_handler(title, instructions, prompt_list)
    m = Message()
    m.add_byte(cMSG_USERAUTH_INFO_RESPONSE)
    m.add_int(len(response_list))
    for r in response_list:
        m.add_string(r)
    self.transport._send_message(m)