import os
from hashlib import sha1, sha256
from paramiko import util
from paramiko.common import DEBUG, byte_chr, byte_ord, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
def _parse_kexdh_gex_request(self, m):
    minbits = m.get_int()
    preferredbits = m.get_int()
    maxbits = m.get_int()
    if preferredbits > self.max_bits:
        preferredbits = self.max_bits
    if preferredbits < self.min_bits:
        preferredbits = self.min_bits
    if minbits > preferredbits:
        minbits = preferredbits
    if maxbits < preferredbits:
        maxbits = preferredbits
    self.min_bits = minbits
    self.preferred_bits = preferredbits
    self.max_bits = maxbits
    pack = self.transport._get_modulus_pack()
    if pack is None:
        raise SSHException("Can't do server-side gex with no modulus pack")
    self.transport._log(DEBUG, 'Picking p ({} <= {} <= {} bits)'.format(minbits, preferredbits, maxbits))
    self.g, self.p = pack.get_modulus(minbits, preferredbits, maxbits)
    m = Message()
    m.add_byte(c_MSG_KEXDH_GEX_GROUP)
    m.add_mpint(self.p)
    m.add_mpint(self.g)
    self.transport._send_message(m)
    self.transport._expect_packet(_MSG_KEXDH_GEX_INIT)