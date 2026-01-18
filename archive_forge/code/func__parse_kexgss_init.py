import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_init(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_INIT message (server mode).

        :param `.Message` m: The content of the SSH2_MSG_KEXGSS_INIT message
        """
    client_token = m.get_string()
    self.e = m.get_mpint()
    if self.e < 1 or self.e > self.P - 1:
        raise SSHException('Client kex "e" is out of range')
    K = pow(self.e, self.x, self.P)
    self.transport.host_key = NullHostKey()
    key = self.transport.host_key.__str__()
    hm = Message()
    hm.add(self.transport.remote_version, self.transport.local_version, self.transport.remote_kex_init, self.transport.local_kex_init)
    hm.add_string(key)
    hm.add_mpint(self.e)
    hm.add_mpint(self.f)
    hm.add_mpint(K)
    H = sha1(hm.asbytes()).digest()
    self.transport._set_K_H(K, H)
    srv_token = self.kexgss.ssh_accept_sec_context(self.gss_host, client_token)
    m = Message()
    if self.kexgss._gss_srv_ctxt_status:
        mic_token = self.kexgss.ssh_get_mic(self.transport.session_id, gss_kex=True)
        m.add_byte(c_MSG_KEXGSS_COMPLETE)
        m.add_mpint(self.f)
        m.add_string(mic_token)
        if srv_token is not None:
            m.add_boolean(True)
            m.add_string(srv_token)
        else:
            m.add_boolean(False)
        self.transport._send_message(m)
        self.transport.gss_kex_used = True
        self.transport._activate_outbound()
    else:
        m.add_byte(c_MSG_KEXGSS_CONTINUE)
        m.add_string(srv_token)
        self.transport._send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE, MSG_KEXGSS_ERROR)