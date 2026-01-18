import random
from dissononce.processing.impl.handshakestate import HandshakeState
from dissononce.extras.processing.handshakestate_guarded import GuardedHandshakeState
from dissononce.extras.processing.handshakestate_switchable import SwitchableHandshakeState
from dissononce.processing.handshakepatterns.handshakepattern import HandshakePattern
from dissononce.processing.handshakepatterns.interactive.IK import IKHandshakePattern
from dissononce.processing.handshakepatterns.interactive.XX import XXHandshakePattern
from dissononce.processing.modifiers.fallback import FallbackPatternModifier
from dissononce.processing.impl.cipherstate import CipherState
from dissononce.cipher.aesgcm import AESGCMCipher
from dissononce.hash.sha256 import SHA256Hash
from dissononce.dh.keypair import KeyPair
from dissononce.dh.x25519.public import PublicKey
from dissononce.dh.private import PrivateKey
from dissononce.dh.x25519.x25519 import X25519DH
from dissononce.extras.dh.dangerous.dh_nogen import NoGenDH
from dissononce.exceptions.decrypt import DecryptFailedException
from google.protobuf.message import DecodeError
from .dissononce_extras.processing.symmetricstate_wa import WASymmetricState
from .proto import wa20_pb2
from .streams.segmented.segmented import SegmentedStream
from .certman.certman import CertMan
from .exceptions.new_rs_exception import NewRemoteStaticException
from .config.client import ClientConfig
from .structs.publickey import PublicKey
from .util.byte import ByteUtil
from.exceptions.handshake_failed_exception import HandshakeFailedException
import logging
def _start_handshake_ik(self, stream, client_payload, s, rs):
    """
        :param stream:
        :type stream: SegmentedStream
        :param s:
        :type s: KeyPair
        :param rs:
        :type rs: PublicKey
        :return:
        :rtype:
        """
    self._handshakestate.initialize(handshake_pattern=IKHandshakePattern(), initiator=True, prologue=self._prologue, s=s, rs=rs)
    message_buffer = bytearray()
    self._handshakestate.write_message(client_payload.SerializeToString(), message_buffer)
    ephemeral_public, static_public, payload = ByteUtil.split(bytes(message_buffer), 32, 48, len(message_buffer) - 32 + 48)
    handshakemessage = wa20_pb2.HandshakeMessage()
    client_hello = wa20_pb2.HandshakeMessage.ClientHello()
    client_hello.ephemeral = ephemeral_public
    client_hello.static = static_public
    client_hello.payload = payload
    handshakemessage.client_hello.MergeFrom(client_hello)
    stream.write_segment(handshakemessage.SerializeToString())
    incoming_handshakemessage = wa20_pb2.HandshakeMessage()
    incoming_handshakemessage.ParseFromString(stream.read_segment())
    if not incoming_handshakemessage.HasField('server_hello'):
        raise HandshakeFailedException('Handshake message does not contain server hello!')
    server_hello = incoming_handshakemessage.server_hello
    if server_hello.HasField('static'):
        raise NewRemoteStaticException(server_hello)
    payload_buffer = bytearray()
    return self._handshakestate.read_message(server_hello.ephemeral + server_hello.static + server_hello.payload, payload_buffer)