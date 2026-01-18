import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_gex_request_old(self, m):
    self.preferred_bits = m.get_int()
    if self.preferred_bits > self.max_bits:
        self.preferred_bits = self.max_bits
    if self.preferred_bits < self.min_bits:
        self.preferred_bits = self.min_bits
    pack = self.transport._get_modulus_pack()
    if pack is None:
        raise SSHException("Can't do server-side gex with no modulus pack")
    self.transport._log(DEBUG, 'Picking p (~ {} bits)'.format(self.preferred_bits))
    self.g, self.p = pack.get_modulus(self.min_bits, self.preferred_bits, self.max_bits)
    m = Message()
    m.add_byte(c_MSG_KEXDH_GEX_GROUP)
    m.add_mpint(self.p)
    m.add_mpint(self.g)
    self.transport._send_message(m)
    self.transport._expect_packet(_MSG_KEXDH_GEX_INIT)
    self.old_style = True