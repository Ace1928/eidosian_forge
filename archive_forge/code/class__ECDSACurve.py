from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
from paramiko.common import four_byte
from paramiko.message import Message
from paramiko.pkey import PKey
from paramiko.ssh_exception import SSHException
from paramiko.util import deflate_long
class _ECDSACurve:
    """
    Represents a specific ECDSA Curve (nistp256, nistp384, etc).

    Handles the generation of the key format identifier and the selection of
    the proper hash function. Also grabs the proper curve from the 'ecdsa'
    package.
    """

    def __init__(self, curve_class, nist_name):
        self.nist_name = nist_name
        self.key_length = curve_class.key_size
        self.key_format_identifier = 'ecdsa-sha2-' + self.nist_name
        if self.key_length <= 256:
            self.hash_object = hashes.SHA256
        elif self.key_length <= 384:
            self.hash_object = hashes.SHA384
        else:
            self.hash_object = hashes.SHA512
        self.curve_class = curve_class