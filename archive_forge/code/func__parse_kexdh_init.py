import os
from hashlib import sha1
from paramiko import util
from paramiko.common import max_byte, zero_byte, byte_chr, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_init(self, m):
    self.e = m.get_mpint()
    if self.e < 1 or self.e > self.P - 1:
        raise SSHException('Client kex "e" is out of range')
    K = pow(self.e, self.x, self.P)
    key = self.transport.get_server_key().asbytes()
    hm = Message()
    hm.add(self.transport.remote_version, self.transport.local_version, self.transport.remote_kex_init, self.transport.local_kex_init)
    hm.add_string(key)
    hm.add_mpint(self.e)
    hm.add_mpint(self.f)
    hm.add_mpint(K)
    H = self.hash_algo(hm.asbytes()).digest()
    self.transport._set_K_H(K, H)
    sig = self.transport.get_server_key().sign_ssh_data(H, self.transport.host_key_type)
    m = Message()
    m.add_byte(c_MSG_KEXDH_REPLY)
    m.add_string(key)
    m.add_mpint(self.f)
    m.add_string(sig)
    self.transport._send_message(m)
    self.transport._activate_outbound()