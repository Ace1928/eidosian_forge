import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_gex_reply(self, m):
    host_key = m.get_string()
    self.f = m.get_mpint()
    sig = m.get_string()
    if self.f < 1 or self.f > self.p - 1:
        raise SSHException('Server kex "f" is out of range')
    K = pow(self.f, self.x, self.p)
    hm = Message()
    hm.add(self.transport.local_version, self.transport.remote_version, self.transport.local_kex_init, self.transport.remote_kex_init, host_key)
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
    self.transport._set_K_H(K, self.hash_algo(hm.asbytes()).digest())
    self.transport._verify_key(host_key, sig)
    self.transport._activate_outbound()