import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
class KexGSSGroup1:
    """
    GSS-API / SSPI Authenticated Diffie-Hellman Key Exchange as defined in `RFC
    4462 Section 2 <https://tools.ietf.org/html/rfc4462.html#section-2>`_
    """
    P = 179769313486231590770839156793787453197860296048756011706444423684197180216158519368947833795864925541502180565485980503646440548199239100050792877003355816639229553136239076508735759914822574862575007425302077447712589550957937778424442426617334727629299387668709205606050270810842907692932019128194467627007
    G = 2
    b7fffffffffffffff = byte_chr(127) + max_byte * 7
    b0000000000000000 = zero_byte * 8
    NAME = 'gss-group1-sha1-toWM5Slw5Ew8Mqkay+al2g=='

    def __init__(self, transport):
        self.transport = transport
        self.kexgss = self.transport.kexgss_ctxt
        self.gss_host = None
        self.x = 0
        self.e = 0
        self.f = 0

    def start_kex(self):
        """
        Start the GSS-API / SSPI Authenticated Diffie-Hellman Key Exchange.
        """
        self._generate_x()
        if self.transport.server_mode:
            self.f = pow(self.G, self.x, self.P)
            self.transport._expect_packet(MSG_KEXGSS_INIT)
            return
        self.e = pow(self.G, self.x, self.P)
        self.gss_host = self.transport.gss_host
        m = Message()
        m.add_byte(c_MSG_KEXGSS_INIT)
        m.add_string(self.kexgss.ssh_init_sec_context(target=self.gss_host))
        m.add_mpint(self.e)
        self.transport._send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_HOSTKEY, MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE, MSG_KEXGSS_ERROR)

    def parse_next(self, ptype, m):
        """
        Parse the next packet.

        :param ptype: The (string) type of the incoming packet
        :param `.Message` m: The packet content
        """
        if self.transport.server_mode and ptype == MSG_KEXGSS_INIT:
            return self._parse_kexgss_init(m)
        elif not self.transport.server_mode and ptype == MSG_KEXGSS_HOSTKEY:
            return self._parse_kexgss_hostkey(m)
        elif self.transport.server_mode and ptype == MSG_KEXGSS_CONTINUE:
            return self._parse_kexgss_continue(m)
        elif not self.transport.server_mode and ptype == MSG_KEXGSS_COMPLETE:
            return self._parse_kexgss_complete(m)
        elif ptype == MSG_KEXGSS_ERROR:
            return self._parse_kexgss_error(m)
        msg = 'GSS KexGroup1 asked to handle packet type {:d}'
        raise SSHException(msg.format(ptype))

    def _generate_x(self):
        """
        generate an "x" (1 < x < q), where q is (p-1)/2.
        p is a 128-byte (1024-bit) number, where the first 64 bits are 1.
        therefore q can be approximated as a 2^1023.  we drop the subset of
        potential x where the first 63 bits are 1, because some of those will
        be larger than q (but this is a tiny tiny subset of potential x).
        """
        while 1:
            x_bytes = os.urandom(128)
            x_bytes = byte_mask(x_bytes[0], 127) + x_bytes[1:]
            first = x_bytes[:8]
            if first not in (self.b7fffffffffffffff, self.b0000000000000000):
                break
        self.x = util.inflate_long(x_bytes)

    def _parse_kexgss_hostkey(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_HOSTKEY message (client mode).

        :param `.Message` m: The content of the SSH2_MSG_KEXGSS_HOSTKEY message
        """
        host_key = m.get_string()
        self.transport.host_key = host_key
        sig = m.get_string()
        self.transport._verify_key(host_key, sig)
        self.transport._expect_packet(MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE)

    def _parse_kexgss_continue(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_CONTINUE message.

        :param `.Message` m: The content of the SSH2_MSG_KEXGSS_CONTINUE
            message
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

    def _parse_kexgss_complete(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_COMPLETE message (client mode).

        :param `.Message` m: The content of the
            SSH2_MSG_KEXGSS_COMPLETE message
        """
        if self.transport.host_key is None:
            self.transport.host_key = NullHostKey()
        self.f = m.get_mpint()
        if self.f < 1 or self.f > self.P - 1:
            raise SSHException('Server kex "f" is out of range')
        mic_token = m.get_string()
        bool = m.get_boolean()
        srv_token = None
        if bool:
            srv_token = m.get_string()
        K = pow(self.f, self.x, self.P)
        hm = Message()
        hm.add(self.transport.local_version, self.transport.remote_version, self.transport.local_kex_init, self.transport.remote_kex_init)
        hm.add_string(self.transport.host_key.__str__())
        hm.add_mpint(self.e)
        hm.add_mpint(self.f)
        hm.add_mpint(K)
        H = sha1(str(hm)).digest()
        self.transport._set_K_H(K, H)
        if srv_token is not None:
            self.kexgss.ssh_init_sec_context(target=self.gss_host, recv_token=srv_token)
            self.kexgss.ssh_check_mic(mic_token, H)
        else:
            self.kexgss.ssh_check_mic(mic_token, H)
        self.transport.gss_kex_used = True
        self.transport._activate_outbound()

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

    def _parse_kexgss_error(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_ERROR message (client mode).
        The server may send a GSS-API error message. if it does, we display
        the error by throwing an exception (client mode).

        :param `.Message` m: The content of the SSH2_MSG_KEXGSS_ERROR message
        :raise SSHException: Contains GSS-API major and minor status as well as
                             the error message and the language tag of the
                             message
        """
        maj_status = m.get_int()
        min_status = m.get_int()
        err_msg = m.get_string()
        m.get_string()
        raise SSHException('GSS-API Error:\nMajor Status: {}\nMinor Status: {}\nError Message: {}\n'.format(maj_status, min_status, err_msg))