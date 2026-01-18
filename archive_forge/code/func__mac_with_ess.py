import struct
import typing
from spnego._ntlm_raw.crypto import RC4Handle, crc32, hmac_md5, rc4
from spnego._ntlm_raw.messages import NegotiateFlags
from spnego.exceptions import OperationNotAvailableError
def _mac_with_ess(flags: int, handle: RC4Handle, signing_key: bytes, seq_num: int, b_data: bytes) -> bytes:
    """NTLM MAC with Extended Session Security

    Generates the NTLM signature when Extended Session Security has been negotiated. The structure of the signature is
    documented at `NTLM signature with ESS`_.

    The algorithm as documented by `MAC with ESS`_ is::

        Define MAC(Handle, SigningKey, SeqNum, Message) as
            Set NTLMSSP_MESSAGE_SIGNATURE.Version to 0x00000001
            Set NTLMSSP_MESSAGE_SIGNATURE.Checksum to HMAC_MD5(SigningKey, ConcatenationOf(SeqNum, Message))[0..7]
            Set NTLMSSP_MESSAGE_SIGNATURE.SeqNum to SeqNum
            Set SeqNum to SeqNum + 1
        EndDefine

        # When NegotiateFlags.key_exch

        Define MAC(Handle, SigningKey, SeqNum, Message) as
            Set NTLMSSP_MESSAGE_SIGNATURE.Version to 0x00000001
            Set NTLMSSP_MESSAGE_SIGNATURE.Checksum to RC4(Handle,
                HMAC_MD5(SigningKey, ConcatenationOf(SeqNum, Message))[0..7])
            Set NTLMSSP_MESSAGE_SIGNATURE.SeqNum to SeqNum
            Set SeqNum to SeqNum + 1
        EndDefine

    Args:
        flags: The negotiated flags between the initiator and acceptor.
        handle: The RC4 handle for the negotiated context.
        signing_key: The key used to sign the message.
        seq_num: The sequence number for the signature.
        b_data: The data/message bytes to sign.

    Returns:
        bytes: The NTLM with ESS signature.

    .. _NTLM signature with ESS:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/2c3b4689-d6f1-4dc6-85c9-0bf01ea34d9f

    .. _MAC with ESS:
        https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-nlmp/a92716d5-d164-4960-9e15-300f4eef44a8
    """
    b_seq_num = struct.pack('<I', seq_num)
    checksum = hmac_md5(signing_key, b_seq_num + b_data)[:8]
    if flags & NegotiateFlags.key_exch:
        checksum = handle.update(checksum)
    return b'\x01\x00\x00\x00' + checksum + b_seq_num