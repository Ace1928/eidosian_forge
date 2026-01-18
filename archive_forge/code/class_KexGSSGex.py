import os
from hashlib import sha1
from paramiko.common import (
from paramiko import util
from paramiko.message import Message
from paramiko.ssh_exception import SSHException
class KexGSSGex:
    """
    GSS-API / SSPI Authenticated Diffie-Hellman Group Exchange as defined in
    `RFC 4462 Section 2 <https://tools.ietf.org/html/rfc4462.html#section-2>`_
    """
    NAME = 'gss-gex-sha1-toWM5Slw5Ew8Mqkay+al2g=='
    min_bits = 1024
    max_bits = 8192
    preferred_bits = 2048

    def __init__(self, transport):
        self.transport = transport
        self.kexgss = self.transport.kexgss_ctxt
        self.gss_host = None
        self.p = None
        self.q = None
        self.g = None
        self.x = None
        self.e = None
        self.f = None
        self.old_style = False

    def start_kex(self):
        """
        Start the GSS-API / SSPI Authenticated Diffie-Hellman Group Exchange
        """
        if self.transport.server_mode:
            self.transport._expect_packet(MSG_KEXGSS_GROUPREQ)
            return
        self.gss_host = self.transport.gss_host
        m = Message()
        m.add_byte(c_MSG_KEXGSS_GROUPREQ)
        m.add_int(self.min_bits)
        m.add_int(self.preferred_bits)
        m.add_int(self.max_bits)
        self.transport._send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_GROUP)

    def parse_next(self, ptype, m):
        """
        Parse the next packet.

        :param ptype: The (string) type of the incoming packet
        :param `.Message` m: The packet content
        """
        if ptype == MSG_KEXGSS_GROUPREQ:
            return self._parse_kexgss_groupreq(m)
        elif ptype == MSG_KEXGSS_GROUP:
            return self._parse_kexgss_group(m)
        elif ptype == MSG_KEXGSS_INIT:
            return self._parse_kexgss_gex_init(m)
        elif ptype == MSG_KEXGSS_HOSTKEY:
            return self._parse_kexgss_hostkey(m)
        elif ptype == MSG_KEXGSS_CONTINUE:
            return self._parse_kexgss_continue(m)
        elif ptype == MSG_KEXGSS_COMPLETE:
            return self._parse_kexgss_complete(m)
        elif ptype == MSG_KEXGSS_ERROR:
            return self._parse_kexgss_error(m)
        msg = 'KexGex asked to handle packet type {:d}'
        raise SSHException(msg.format(ptype))

    def _generate_x(self):
        q = (self.p - 1) // 2
        qnorm = util.deflate_long(q, 0)
        qhbyte = byte_ord(qnorm[0])
        byte_count = len(qnorm)
        qmask = 255
        while not qhbyte & 128:
            qhbyte <<= 1
            qmask >>= 1
        while True:
            x_bytes = os.urandom(byte_count)
            x_bytes = byte_mask(x_bytes[0], qmask) + x_bytes[1:]
            x = util.inflate_long(x_bytes, 1)
            if x > 1 and x < q:
                break
        self.x = x

    def _parse_kexgss_groupreq(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_GROUPREQ message (server mode).

        :param `.Message` m: The content of the
            SSH2_MSG_KEXGSS_GROUPREQ message
        """
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
        m.add_byte(c_MSG_KEXGSS_GROUP)
        m.add_mpint(self.p)
        m.add_mpint(self.g)
        self.transport._send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_INIT)

    def _parse_kexgss_group(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_GROUP message (client mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_GROUP message
        """
        self.p = m.get_mpint()
        self.g = m.get_mpint()
        bitlen = util.bit_length(self.p)
        if bitlen < 1024 or bitlen > 8192:
            raise SSHException("Server-generated gex p (don't ask) is out of range ({} bits)".format(bitlen))
        self.transport._log(DEBUG, 'Got server p ({} bits)'.format(bitlen))
        self._generate_x()
        self.e = pow(self.g, self.x, self.p)
        m = Message()
        m.add_byte(c_MSG_KEXGSS_INIT)
        m.add_string(self.kexgss.ssh_init_sec_context(target=self.gss_host))
        m.add_mpint(self.e)
        self.transport._send_message(m)
        self.transport._expect_packet(MSG_KEXGSS_HOSTKEY, MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE, MSG_KEXGSS_ERROR)

    def _parse_kexgss_gex_init(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_INIT message (server mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_INIT message
        """
        client_token = m.get_string()
        self.e = m.get_mpint()
        if self.e < 1 or self.e > self.p - 1:
            raise SSHException('Client kex "e" is out of range')
        self._generate_x()
        self.f = pow(self.g, self.x, self.p)
        K = pow(self.e, self.x, self.p)
        self.transport.host_key = NullHostKey()
        key = self.transport.host_key.__str__()
        hm = Message()
        hm.add(self.transport.remote_version, self.transport.local_version, self.transport.remote_kex_init, self.transport.local_kex_init, key)
        hm.add_int(self.min_bits)
        hm.add_int(self.preferred_bits)
        hm.add_int(self.max_bits)
        hm.add_mpint(self.p)
        hm.add_mpint(self.g)
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

    def _parse_kexgss_hostkey(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_HOSTKEY message (client mode).

        :param `Message` m: The content of the SSH2_MSG_KEXGSS_HOSTKEY message
        """
        host_key = m.get_string()
        self.transport.host_key = host_key
        sig = m.get_string()
        self.transport._verify_key(host_key, sig)
        self.transport._expect_packet(MSG_KEXGSS_CONTINUE, MSG_KEXGSS_COMPLETE)

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

    def _parse_kexgss_error(self, m):
        """
        Parse the SSH2_MSG_KEXGSS_ERROR message (client mode).
        The server may send a GSS-API error message. if it does, we display
        the error by throwing an exception (client mode).

        :param `Message` m:  The content of the SSH2_MSG_KEXGSS_ERROR message
        :raise SSHException: Contains GSS-API major and minor status as well as
                             the error message and the language tag of the
                             message
        """
        maj_status = m.get_int()
        min_status = m.get_int()
        err_msg = m.get_string()
        m.get_string()
        raise SSHException('GSS-API Error:\nMajor Status: {}\nMinor Status: {}\nError Message: {}\n'.format(maj_status, min_status, err_msg))