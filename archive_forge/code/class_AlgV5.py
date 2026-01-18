import hashlib
import secrets
import struct
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Tuple, Union, cast
from pypdf._crypt_providers import (
from ._utils import b_, logger_warning
from .generic import (
class AlgV5:

    @staticmethod
    def verify_owner_password(R: int, password: bytes, o_value: bytes, oe_value: bytes, u_value: bytes) -> bytes:
        """
        Algorithm 3.2a Computing an encryption key.

        To understand the algorithm below, it is necessary to treat the O and U
        strings in the Encrypt dictionary as made up of three sections.
        The first 32 bytes are a hash value (explained below). The next 8 bytes
        are called the Validation Salt. The final 8 bytes are called the Key Salt.

        1. The password string is generated from Unicode input by processing the
           input string with the SASLprep (IETF RFC 4013) profile of
           stringprep (IETF RFC 3454), and then converting to a UTF-8
           representation.
        2. Truncate the UTF-8 representation to 127 bytes if it is longer than
           127 bytes.
        3. Test the password against the owner key by computing the SHA-256 hash
           of the UTF-8 password concatenated with the 8 bytes of owner
           Validation Salt, concatenated with the 48-byte U string. If the
           32-byte result matches the first 32 bytes of the O string, this is
           the owner password.
           Compute an intermediate owner key by computing the SHA-256 hash of
           the UTF-8 password concatenated with the 8 bytes of owner Key Salt,
           concatenated with the 48-byte U string. The 32-byte result is the
           key used to decrypt the 32-byte OE string using AES-256 in CBC mode
           with no padding and an initialization vector of zero.
           The 32-byte result is the file encryption key.
        4. Test the password against the user key by computing the SHA-256 hash
           of the UTF-8 password concatenated with the 8 bytes of user
           Validation Salt. If the 32 byte result matches the first 32 bytes of
           the U string, this is the user password.
           Compute an intermediate user key by computing the SHA-256 hash of the
           UTF-8 password concatenated with the 8 bytes of user Key Salt.
           The 32-byte result is the key used to decrypt the 32-byte
           UE string using AES-256 in CBC mode with no padding and an
           initialization vector of zero. The 32-byte result is the file
           encryption key.
        5. Decrypt the 16-byte Perms string using AES-256 in ECB mode with an
           initialization vector of zero and the file encryption key as the key.
           Verify that bytes 9-11 of the result are the characters ‘a’, ‘d’, ‘b’.
           Bytes 0-3 of the decrypted Perms entry, treated as a little-endian
           integer, are the user permissions.
           They should match the value in the P key.

        Args:
            R: A number specifying which revision of the standard security
                handler shall be used to interpret this dictionary
            password: The owner password
            o_value: A 32-byte string, based on both the owner and user passwords,
                that shall be used in computing the encryption key and in
                determining whether a valid owner password was entered
            oe_value:
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password and,
                if so, whether a valid user or owner password was entered.

        Returns:
            The key
        """
        password = password[:127]
        if AlgV5.calculate_hash(R, password, o_value[32:40], u_value[:48]) != o_value[:32]:
            return b''
        iv = bytes((0 for _ in range(16)))
        tmp_key = AlgV5.calculate_hash(R, password, o_value[40:48], u_value[:48])
        key = aes_cbc_decrypt(tmp_key, iv, oe_value)
        return key

    @staticmethod
    def verify_user_password(R: int, password: bytes, u_value: bytes, ue_value: bytes) -> bytes:
        """
        See :func:`verify_owner_password`.

        Args:
            R: A number specifying which revision of the standard security
                handler shall be used to interpret this dictionary
            password: The user password
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password
                and, if so, whether a valid user or owner password was entered.
            ue_value:

        Returns:
            bytes
        """
        password = password[:127]
        if AlgV5.calculate_hash(R, password, u_value[32:40], b'') != u_value[:32]:
            return b''
        iv = bytes((0 for _ in range(16)))
        tmp_key = AlgV5.calculate_hash(R, password, u_value[40:48], b'')
        return aes_cbc_decrypt(tmp_key, iv, ue_value)

    @staticmethod
    def calculate_hash(R: int, password: bytes, salt: bytes, udata: bytes) -> bytes:
        k = hashlib.sha256(password + salt + udata).digest()
        if R < 6:
            return k
        count = 0
        while True:
            count += 1
            k1 = password + k + udata
            e = aes_cbc_encrypt(k[:16], k[16:32], k1 * 64)
            hash_fn = (hashlib.sha256, hashlib.sha384, hashlib.sha512)[sum(e[:16]) % 3]
            k = hash_fn(e).digest()
            if count >= 64 and e[-1] <= count - 32:
                break
        return k[:32]

    @staticmethod
    def verify_perms(key: bytes, perms: bytes, p: int, metadata_encrypted: bool) -> bool:
        """
        See :func:`verify_owner_password` and :func:`compute_perms_value`.

        Args:
            key: The owner password
            perms:
            p: A set of flags specifying which operations shall be permitted
                when the document is opened with user access.
                If bit 2 is set to 1, all other bits are ignored and all
                operations are permitted.
                If bit 2 is set to 0, permission for operations are based on
                the values of the remaining flags defined in Table 24.
            metadata_encrypted:

        Returns:
            A boolean
        """
        b8 = b'T' if metadata_encrypted else b'F'
        p1 = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb'
        p2 = aes_ecb_decrypt(key, perms)
        return p1 == p2[:12]

    @staticmethod
    def generate_values(R: int, user_password: bytes, owner_password: bytes, key: bytes, p: int, metadata_encrypted: bool) -> Dict[Any, Any]:
        user_password = user_password[:127]
        owner_password = owner_password[:127]
        u_value, ue_value = AlgV5.compute_U_value(R, user_password, key)
        o_value, oe_value = AlgV5.compute_O_value(R, owner_password, key, u_value)
        perms = AlgV5.compute_Perms_value(key, p, metadata_encrypted)
        return {'/U': u_value, '/UE': ue_value, '/O': o_value, '/OE': oe_value, '/Perms': perms}

    @staticmethod
    def compute_U_value(R: int, password: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """
        Algorithm 3.8 Computing the encryption dictionary’s U (user password)
        and UE (user encryption key) values.

        1. Generate 16 random bytes of data using a strong random number generator.
           The first 8 bytes are the User Validation Salt. The second 8 bytes
           are the User Key Salt. Compute the 32-byte SHA-256 hash of the
           password concatenated with the User Validation Salt. The 48-byte
           string consisting of the 32-byte hash followed by the User
           Validation Salt followed by the User Key Salt is stored as the U key.
        2. Compute the 32-byte SHA-256 hash of the password concatenated with
           the User Key Salt. Using this hash as the key, encrypt the file
           encryption key using AES-256 in CBC mode with no padding and an
           initialization vector of zero. The resulting 32-byte string is stored
           as the UE key.

        Args:
            R:
            password:
            key:

        Returns:
            A tuple (u-value, ue value)
        """
        random_bytes = secrets.token_bytes(16)
        val_salt = random_bytes[:8]
        key_salt = random_bytes[8:]
        u_value = AlgV5.calculate_hash(R, password, val_salt, b'') + val_salt + key_salt
        tmp_key = AlgV5.calculate_hash(R, password, key_salt, b'')
        iv = bytes((0 for _ in range(16)))
        ue_value = aes_cbc_encrypt(tmp_key, iv, key)
        return (u_value, ue_value)

    @staticmethod
    def compute_O_value(R: int, password: bytes, key: bytes, u_value: bytes) -> Tuple[bytes, bytes]:
        """
        Algorithm 3.9 Computing the encryption dictionary’s O (owner password)
        and OE (owner encryption key) values.

        1. Generate 16 random bytes of data using a strong random number
           generator. The first 8 bytes are the Owner Validation Salt. The
           second 8 bytes are the Owner Key Salt. Compute the 32-byte SHA-256
           hash of the password concatenated with the Owner Validation Salt and
           then concatenated with the 48-byte U string as generated in
           Algorithm 3.8. The 48-byte string consisting of the 32-byte hash
           followed by the Owner Validation Salt followed by the Owner Key Salt
           is stored as the O key.
        2. Compute the 32-byte SHA-256 hash of the password concatenated with
           the Owner Key Salt and then concatenated with the 48-byte U string as
           generated in Algorithm 3.8. Using this hash as the key,
           encrypt the file encryption key using AES-256 in CBC mode with
           no padding and an initialization vector of zero.
           The resulting 32-byte string is stored as the OE key.

        Args:
            R:
            password:
            key:
            u_value: A 32-byte string, based on the user password, that shall be
                used in determining whether to prompt the user for a password
                and, if so, whether a valid user or owner password was entered.

        Returns:
            A tuple (O value, OE value)
        """
        random_bytes = secrets.token_bytes(16)
        val_salt = random_bytes[:8]
        key_salt = random_bytes[8:]
        o_value = AlgV5.calculate_hash(R, password, val_salt, u_value) + val_salt + key_salt
        tmp_key = AlgV5.calculate_hash(R, password, key_salt, u_value[:48])
        iv = bytes((0 for _ in range(16)))
        oe_value = aes_cbc_encrypt(tmp_key, iv, key)
        return (o_value, oe_value)

    @staticmethod
    def compute_Perms_value(key: bytes, p: int, metadata_encrypted: bool) -> bytes:
        """
        Algorithm 3.10 Computing the encryption dictionary’s Perms
        (permissions) value.

        1. Extend the permissions (contents of the P integer) to 64 bits by
           setting the upper 32 bits to all 1’s.
           (This allows for future extension without changing the format.)
        2. Record the 8 bytes of permission in the bytes 0-7 of the block,
           low order byte first.
        3. Set byte 8 to the ASCII value ' T ' or ' F ' according to the
           EncryptMetadata Boolean.
        4. Set bytes 9-11 to the ASCII characters ' a ', ' d ', ' b '.
        5. Set bytes 12-15 to 4 bytes of random data, which will be ignored.
        6. Encrypt the 16-byte block using AES-256 in ECB mode with an
           initialization vector of zero, using the file encryption key as the
           key. The result (16 bytes) is stored as the Perms string, and checked
           for validity when the file is opened.

        Args:
            key:
            p: A set of flags specifying which operations shall be permitted
                when the document is opened with user access. If bit 2 is set to 1,
                all other bits are ignored and all operations are permitted.
                If bit 2 is set to 0, permission for operations are based on the
                values of the remaining flags defined in Table 24.
            metadata_encrypted: A boolean indicating if the metadata is encrypted.

        Returns:
            The perms value
        """
        b8 = b'T' if metadata_encrypted else b'F'
        rr = secrets.token_bytes(4)
        data = struct.pack('<I', p) + b'\xff\xff\xff\xff' + b8 + b'adb' + rr
        perms = aes_ecb_encrypt(key, data)
        return perms