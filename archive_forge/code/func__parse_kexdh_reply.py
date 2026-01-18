import os
from hashlib import sha1
from paramiko import util
from paramiko.common import max_byte, zero_byte, byte_chr, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_reply(self, m):
    host_key = m.get_string()
    self.f = m.get_mpint()
    if self.f < 1 or self.f > self.P - 1:
        raise SSHException('Server kex "f" is out of range')
    sig = m.get_binary()
    K = pow(self.f, self.x, self.P)
    hm = Message()
    hm.add(self.transport.local_version, self.transport.remote_version, self.transport.local_kex_init, self.transport.remote_kex_init)
    hm.add_string(host_key)
    hm.add_mpint(self.e)
    hm.add_mpint(self.f)
    hm.add_mpint(K)
    self.transport._set_K_H(K, self.hash_algo(hm.asbytes()).digest())
    self.transport._verify_key(host_key, sig)
    self.transport._activate_outbound()