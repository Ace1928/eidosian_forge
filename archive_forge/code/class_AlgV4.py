import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
class AlgV4:

    @staticmethod
    def compute_key(password: bytes, rev: int, key_size: int, o_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        """
        Algorithm 2: Computing an encryption key.

        a) Pad or truncate the password string to exactly 32 bytes. If the
           password string is more than 32 bytes long,
           use only its first 32 bytes; if it is less than 32 bytes long, pad it
           by appending the required number of
           additional bytes from the beginning of the following padding string:
                < 28 BF 4E 5E 4E 75 8A 41 64 00 4E 56 FF FA 01 08
                2E 2E 00 B6 D0 68 3E 80 2F 0C A9 FE 64 53 69 7A >
           That is, if the password string is n bytes long, append
           the first 32 - n bytes of the padding string to the end
           of the password string. If the password string is empty
           (zero-length), meaning there is no user password,
           substitute the entire padding string in its place.

        b) Initialize the MD5 hash function and pass the result of step (a)
           as input to this function.
        c) Pass the value of the encryption dictionary’s O entry to the
           MD5 hash function. ("Algorithm 3: Computing
           the encryption dictionary’s O (owner password) value" shows how the
           O value is computed.)
        d) Convert the integer value of the P entry to a 32-bit unsigned binary
           number and pass these bytes to the
           MD5 hash function, low-order byte first.
        e) Pass the first element of the file’s file identifier array (the value
           of the ID entry in the document’s trailer
           dictionary; see Table 15) to the MD5 hash function.
        f) (Security handlers of revision 4 or greater) If document metadata is
           not being encrypted, pass 4 bytes with
           the value 0xFFFFFFFF to the MD5 hash function.
        g) Finish the hash.
        h) (Security handlers of revision 3 or greater) Do the following
           50 times: Take the output from the previous
           MD5 hash and pass the first n bytes of the output as input into a new
           MD5 hash, where n is the number of
           bytes of the encryption key as defined by the value of the encryption
           dictionary’s Length entry.
        i) Set the encryption key to the first n bytes of the output from the
           final MD5 hash, where n shall always be 5
           for security handlers of revision 2 but, for security handlers of
           revision 3 or greater, shall depend on the
           value of the encryption dictionary’s Length entry.

        Args:
            password: The encryption secret as a bytes-string
            rev: The encryption revision (see PDF standard)
            key_size: The size of the key in bytes
            o_entry: The owner entry
            P: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            id1_entry:
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            The u_hash digest of length key_size
        """
        a = _padding(password)
        u_hash = hashlib.md5(a)
        u_hash.update(o_entry)
        u_hash.update(struct.pack('<I', P))
        u_hash.update(id1_entry)
        if rev >= 4 and (not metadata_encrypted):
            u_hash.update(b'\xff\xff\xff\xff')
        u_hash_digest = u_hash.digest()
        length = key_size // 8
        if rev >= 3:
            for _ in range(50):
                u_hash_digest = hashlib.md5(u_hash_digest[:length]).digest()
        return u_hash_digest[:length]

    @staticmethod
    def compute_O_value_key(owner_password: bytes, rev: int, key_size: int) -> bytes:
        """
        Algorithm 3: Computing the encryption dictionary’s O (owner password) value.

        a) Pad or truncate the owner password string as described in step (a)
           of "Algorithm 2: Computing an encryption key".
           If there is no owner password, use the user password instead.
        b) Initialize the MD5 hash function and pass the result of step (a) as
           input to this function.
        c) (Security handlers of revision 3 or greater) Do the following 50 times:
           Take the output from the previous
           MD5 hash and pass it as input into a new MD5 hash.
        d) Create an RC4 encryption key using the first n bytes of the output
           from the final MD5 hash, where n shall
           always be 5 for security handlers of revision 2 but, for security
           handlers of revision 3 or greater, shall
           depend on the value of the encryption dictionary’s Length entry.
        e) Pad or truncate the user password string as described in step (a) of
           "Algorithm 2: Computing an encryption key".
        f) Encrypt the result of step (e), using an RC4 encryption function with
           the encryption key obtained in step (d).
        g) (Security handlers of revision 3 or greater) Do the following 19 times:
           Take the output from the previous
           invocation of the RC4 function and pass it as input to a new
           invocation of the function; use an encryption
           key generated by taking each byte of the encryption key obtained in
           step (d) and performing an XOR
           (exclusive or) operation between that byte and the single-byte value
           of the iteration counter (from 1 to 19).
        h) Store the output from the final invocation of the RC4 function as
           the value of the O entry in the encryption dictionary.

        Args:
            owner_password:
            rev: The encryption revision (see PDF standard)
            key_size: The size of the key in bytes

        Returns:
            The RC4 key
        """
        a = _padding(owner_password)
        o_hash_digest = hashlib.md5(a).digest()
        if rev >= 3:
            for _ in range(50):
                o_hash_digest = hashlib.md5(o_hash_digest).digest()
        rc4_key = o_hash_digest[:key_size // 8]
        return rc4_key

    @staticmethod
    def compute_O_value(rc4_key: bytes, user_password: bytes, rev: int) -> bytes:
        """
        See :func:`compute_O_value_key`.

        Args:
            rc4_key:
            user_password:
            rev: The encryption revision (see PDF standard)

        Returns:
            The RC4 encrypted
        """
        a = _padding(user_password)
        rc4_enc = rc4_encrypt(rc4_key, a)
        if rev >= 3:
            for i in range(1, 20):
                key = bytes(bytearray((x ^ i for x in rc4_key)))
                rc4_enc = rc4_encrypt(key, rc4_enc)
        return rc4_enc

    @staticmethod
    def compute_U_value(key: bytes, rev: int, id1_entry: bytes) -> bytes:
        """
        Algorithm 4: Computing the encryption dictionary’s U (user password) value.

        (Security handlers of revision 2)

        a) Create an encryption key based on the user password string, as
           described in "Algorithm 2: Computing an encryption key".
        b) Encrypt the 32-byte padding string shown in step (a) of
           "Algorithm 2: Computing an encryption key", using an RC4 encryption
           function with the encryption key from the preceding step.
        c) Store the result of step (b) as the value of the U entry in the
           encryption dictionary.

        Args:
            key:
            rev: The encryption revision (see PDF standard)
            id1_entry:

        Returns:
            The value
        """
        if rev <= 2:
            value = rc4_encrypt(key, _PADDING)
            return value
        '\n        Algorithm 5: Computing the encryption dictionary’s U (user password) value.\n\n        (Security handlers of revision 3 or greater)\n\n        a) Create an encryption key based on the user password string, as\n           described in "Algorithm 2: Computing an encryption key".\n        b) Initialize the MD5 hash function and pass the 32-byte padding string\n           shown in step (a) of "Algorithm 2:\n           Computing an encryption key" as input to this function.\n        c) Pass the first element of the file’s file identifier array (the value\n           of the ID entry in the document’s trailer\n           dictionary; see Table 15) to the hash function and finish the hash.\n        d) Encrypt the 16-byte result of the hash, using an RC4 encryption\n           function with the encryption key from step (a).\n        e) Do the following 19 times: Take the output from the previous\n           invocation of the RC4 function and pass it as input to a new\n           invocation of the function; use an encryption key generated by\n           taking each byte of the original encryption key obtained in\n           step (a) and performing an XOR (exclusive or) operation between that\n           byte and the single-byte value of the iteration counter (from 1 to 19).\n        f) Append 16 bytes of arbitrary padding to the output from the final\n           invocation of the RC4 function and store the 32-byte result as the\n           value of the U entry in the encryption dictionary.\n        '
        u_hash = hashlib.md5(_PADDING)
        u_hash.update(id1_entry)
        rc4_enc = rc4_encrypt(key, u_hash.digest())
        for i in range(1, 20):
            rc4_key = bytes(bytearray((x ^ i for x in key)))
            rc4_enc = rc4_encrypt(rc4_key, rc4_enc)
        return _padding(rc4_enc)

    @staticmethod
    def verify_user_password(user_password: bytes, rev: int, key_size: int, o_entry: bytes, u_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        """
        Algorithm 6: Authenticating the user password.

        a) Perform all but the last step of "Algorithm 4: Computing the
           encryption dictionary’s U (user password) value (Security handlers of
           revision 2)" or "Algorithm 5: Computing the encryption dictionary’s U
           (user password) value (Security handlers of revision 3 or greater)"
           using the supplied password string.
        b) If the result of step (a) is equal to the value of the encryption
           dictionary’s U entry (comparing on the first 16 bytes in the case of
           security handlers of revision 3 or greater), the password supplied is
           the correct user password. The key obtained in step (a) (that is, in
           the first step of "Algorithm 4: Computing the encryption
           dictionary’s U (user password) value
           (Security handlers of revision 2)" or
           "Algorithm 5: Computing the encryption dictionary’s U (user password)
           value (Security handlers of revision 3 or greater)") shall be used
           to decrypt the document.

        Args:
            user_password: The user password as a bytes stream
            rev: The encryption revision (see PDF standard)
            key_size: The size of the key in bytes
            o_entry: The owner entry
            u_entry: The user entry
            P: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            id1_entry:
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            The key
        """
        key = AlgV4.compute_key(user_password, rev, key_size, o_entry, P, id1_entry, metadata_encrypted)
        u_value = AlgV4.compute_U_value(key, rev, id1_entry)
        if rev >= 3:
            u_value = u_value[:16]
            u_entry = u_entry[:16]
        if u_value != u_entry:
            key = b''
        return key

    @staticmethod
    def verify_owner_password(owner_password: bytes, rev: int, key_size: int, o_entry: bytes, u_entry: bytes, P: int, id1_entry: bytes, metadata_encrypted: bool) -> bytes:
        """
        Algorithm 7: Authenticating the owner password.

        a) Compute an encryption key from the supplied password string, as
           described in steps (a) to (d) of
           "Algorithm 3: Computing the encryption dictionary’s O (owner password)
           value".
        b) (Security handlers of revision 2 only) Decrypt the value of the
           encryption dictionary’s O entry, using an RC4
           encryption function with the encryption key computed in step (a).
           (Security handlers of revision 3 or greater) Do the following 20 times:
           Decrypt the value of the encryption dictionary’s O entry (first iteration)
           or the output from the previous iteration (all subsequent iterations),
           using an RC4 encryption function with a different encryption key at
           each iteration. The key shall be generated by taking the original key
           (obtained in step (a)) and performing an XOR (exclusive or) operation
           between each byte of the key and the single-byte value of the
           iteration counter (from 19 to 0).
        c) The result of step (b) purports to be the user password.
           Authenticate this user password using
           "Algorithm 6: Authenticating the user password".
           If it is correct, the password supplied is the correct owner password.

        Args:
            owner_password:
            rev: The encryption revision (see PDF standard)
            key_size: The size of the key in bytes
            o_entry: The owner entry
            u_entry: The user entry
            P: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            id1_entry:
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            bytes
        """
        rc4_key = AlgV4.compute_O_value_key(owner_password, rev, key_size)
        if rev <= 2:
            user_password = rc4_decrypt(rc4_key, o_entry)
        else:
            user_password = o_entry
            for i in range(19, -1, -1):
                key = bytes(bytearray((x ^ i for x in rc4_key)))
                user_password = rc4_decrypt(key, user_password)
        return AlgV4.verify_user_password(user_password, rev, key_size, o_entry, u_entry, P, id1_entry, metadata_encrypted)