import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexgss_complete(self, m):
    """
        Parse the SSH2_MSG_KEXGSS_COMPLETE message (client mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_COMPLETE message
        """
    if self.transport.host_key is None:
        self.transport.host_key = NullHostKey()
    self.f = m.get_mpint()
    mic_token = m.get_string()
    bool = m.get_boolean()
    srv_token = None
    if bool:
        srv_token = m.get_string()
    if self.f < 1 or self.f > self.p - 1:
        raise SSHException('Server kex "f" is out of range')
    K = pow(self.f, self.x, self.p)
    hm = Message()
    hm.add(self.transport.local_version, self.transport.remote_version, self.transport.local_kex_init, self.transport.remote_kex_init, self.transport.host_key.__str__())
    if not self.old_style:
        hm.add_int(self.min_bits)
    hm.add_int(self.preferred_bits)
    if not self.old_style:
        hm.add_int(self.max_bits)
    hm.add_mpint(self.p)
    hm.add_mpint(self.g)
    hm.add_mpint(self.e)
    hm.add_mpint(self.f)
    hm.add_mpint(K)
    H = sha1(hm.asbytes()).digest()
    self.transport._set_K_H(K, H)
    if srv_token is not None:
        self.kexgss.ssh_init_sec_context(target=self.gss_host, recv_token=srv_token)
        self.kexgss.ssh_check_mic(mic_token, H)
    else:
        self.kexgss.ssh_check_mic(mic_token, H)
    self.transport.gss_kex_used = True
    self.transport._activate_outbound()