import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_continue(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_CONTINUE message.

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_CONTINUE message
        """
    if not self.transport.server_mode:
        srv_token = m.get_string()
        m = Message()
        m.add_byte(c_MSG_KEXGSS_CONTINUE)
        m.add_string(self.kexgss.ssh_init_sec_context(target=self.gss_host, recv_token=srv_token))
        self.transport.send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE, MSG_KEXGSS_ERROR)
    else:
        pass