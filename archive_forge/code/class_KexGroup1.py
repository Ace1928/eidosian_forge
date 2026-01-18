import os
from hashlib import sha1
from paramiko import util
from paramiko.common import max_byte, zero_byte, byte_chr, byte_mask
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
class KexGroup1:
    P = 179769313486231590770839156793787453197860296048756011706444423684197180216158519368947833795864925541502180565485980503646440548199239100050792877003355816639229553136239076508735759914822574862575007425302077447712589550957937778424442426617334727629299387668709205606050270810842907692932019128194467627007
    G = 2
    name = 'diffie-hellman-group1-sha1'
    hash_algo = sha1

    def __init__(self, transport):
        self.transport = transport
        self.x = 0
        self.e = 0
        self.f = 0

    def start_kex(self):
        self._generate_x()
        if self.transport.server_mode:
            self.f = pow(self.G, self.x, self.P)
            self.transport._expect_packet(_MSG_KEXDH_INIT)
            return
        self.e = pow(self.G, self.x, self.P)
        m = Message()
        m.add_byte(c_MSG_KEXDH_INIT)
        m.add_mpint(self.e)
        self.transport._send_message(m)
        self.transport._expect_packet(_MSG_KEXDH_REPLY)

    def parse_next(self, ptype, m):
        if self.transport.server_mode and ptype == _MSG_KEXDH_INIT:
            return self._parse_kexdh_init(m)
        elif not self.transport.server_mode and ptype == _MSG_KEXDH_REPLY:
            return self._parse_kexdh_reply(m)
        msg = 'KexGroup1 asked to handle packet type {:d}'
        raise SSHException(msg.format(ptype))

    def _generate_x(self):
        while 1:
            x_bytes = os.urandom(128)
            x_bytes = byte_mask(x_bytes[0], 127) + x_bytes[1:]
            if x_bytes[:8] != b7fffffffffffffff and x_bytes[:8] != b0000000000000000:
                break
        self.x = util.inflate_long(x_bytes)

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