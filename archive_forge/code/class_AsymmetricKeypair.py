from __future__ import absolute_import, division, print_function
import os
from base64 import b64encode, b64decode
from getpass import getuser
from socket import gethostname
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
class AsymmetricKeypair(object):
    """Container for newly generated asymmetric key pairs or those loaded from existing files"""

    @classmethod
    def generate(cls, keytype='rsa', size=None, passphrase=None):
        """Returns an Asymmetric_Keypair object generated with the supplied parameters
           or defaults to an unencrypted RSA-2048 key

           :keytype: One of rsa, dsa, ecdsa, ed25519
           :size: The key length for newly generated keys
           :passphrase: Secret of type Bytes used to encrypt the private key being generated
        """
        if keytype not in _ALGORITHM_PARAMETERS.keys():
            raise InvalidKeyTypeError('%s is not a valid keytype. Valid keytypes are %s' % (keytype, ', '.join(_ALGORITHM_PARAMETERS.keys())))
        if not size:
            size = _ALGORITHM_PARAMETERS[keytype]['default_size']
        elif size not in _ALGORITHM_PARAMETERS[keytype]['valid_sizes']:
            raise InvalidKeySizeError('%s is not a valid key size for %s keys' % (size, keytype))
        if passphrase:
            encryption_algorithm = get_encryption_algorithm(passphrase)
        else:
            encryption_algorithm = serialization.NoEncryption()
        if keytype == 'rsa':
            privatekey = rsa.generate_private_key(public_exponent=65537, key_size=size, backend=backend)
        elif keytype == 'dsa':
            privatekey = dsa.generate_private_key(key_size=size, backend=backend)
        elif keytype == 'ed25519':
            privatekey = Ed25519PrivateKey.generate()
        elif keytype == 'ecdsa':
            privatekey = ec.generate_private_key(_ALGORITHM_PARAMETERS['ecdsa']['curves'][size], backend=backend)
        publickey = privatekey.public_key()
        return cls(keytype=keytype, size=size, privatekey=privatekey, publickey=publickey, encryption_algorithm=encryption_algorithm)

    @classmethod
    def load(cls, path, passphrase=None, private_key_format='PEM', public_key_format='PEM', no_public_key=False):
        """Returns an Asymmetric_Keypair object loaded from the supplied file path

           :path: A path to an existing private key to be loaded
           :passphrase: Secret of type bytes used to decrypt the private key being loaded
           :private_key_format: Format of private key to be loaded
           :public_key_format: Format of public key to be loaded
           :no_public_key: Set 'True' to only load a private key and automatically populate the matching public key
        """
        if passphrase:
            encryption_algorithm = get_encryption_algorithm(passphrase)
        else:
            encryption_algorithm = serialization.NoEncryption()
        privatekey = load_privatekey(path, passphrase, private_key_format)
        if no_public_key:
            publickey = privatekey.public_key()
        else:
            publickey = load_publickey(path + '.pub', public_key_format)
        if isinstance(privatekey, Ed25519PrivateKey):
            size = _ALGORITHM_PARAMETERS['ed25519']['default_size']
        else:
            size = privatekey.key_size
        if isinstance(privatekey, rsa.RSAPrivateKey):
            keytype = 'rsa'
        elif isinstance(privatekey, dsa.DSAPrivateKey):
            keytype = 'dsa'
        elif isinstance(privatekey, ec.EllipticCurvePrivateKey):
            keytype = 'ecdsa'
        elif isinstance(privatekey, Ed25519PrivateKey):
            keytype = 'ed25519'
        else:
            raise InvalidKeyTypeError("Key type '%s' is not supported" % type(privatekey))
        return cls(keytype=keytype, size=size, privatekey=privatekey, publickey=publickey, encryption_algorithm=encryption_algorithm)

    def __init__(self, keytype, size, privatekey, publickey, encryption_algorithm):
        """
           :keytype: One of rsa, dsa, ecdsa, ed25519
           :size: The key length for the private key of this key pair
           :privatekey: Private key object of this key pair
           :publickey: Public key object of this key pair
           :encryption_algorithm: Hashed secret used to encrypt the private key of this key pair
        """
        self.__size = size
        self.__keytype = keytype
        self.__privatekey = privatekey
        self.__publickey = publickey
        self.__encryption_algorithm = encryption_algorithm
        try:
            self.verify(self.sign(b'message'), b'message')
        except InvalidSignatureError:
            raise InvalidPublicKeyFileError('The private key and public key of this keypair do not match')

    def __eq__(self, other):
        if not isinstance(other, AsymmetricKeypair):
            return NotImplemented
        return compare_publickeys(self.public_key, other.public_key) and compare_encryption_algorithms(self.encryption_algorithm, other.encryption_algorithm)

    def __ne__(self, other):
        return not self == other

    @property
    def private_key(self):
        """Returns the private key of this key pair"""
        return self.__privatekey

    @property
    def public_key(self):
        """Returns the public key of this key pair"""
        return self.__publickey

    @property
    def size(self):
        """Returns the size of the private key of this key pair"""
        return self.__size

    @property
    def key_type(self):
        """Returns the key type of this key pair"""
        return self.__keytype

    @property
    def encryption_algorithm(self):
        """Returns the key encryption algorithm of this key pair"""
        return self.__encryption_algorithm

    def sign(self, data):
        """Returns signature of data signed with the private key of this key pair

           :data: byteslike data to sign
        """
        try:
            signature = self.__privatekey.sign(data, **_ALGORITHM_PARAMETERS[self.__keytype]['signer_params'])
        except TypeError as e:
            raise InvalidDataError(e)
        return signature

    def verify(self, signature, data):
        """Verifies that the signature associated with the provided data was signed
           by the private key of this key pair.

           :signature: signature to verify
           :data: byteslike data signed by the provided signature
        """
        try:
            return self.__publickey.verify(signature, data, **_ALGORITHM_PARAMETERS[self.__keytype]['signer_params'])
        except InvalidSignature:
            raise InvalidSignatureError

    def update_passphrase(self, passphrase=None):
        """Updates the encryption algorithm of this key pair

           :passphrase: Byte secret used to encrypt this key pair
        """
        if passphrase:
            self.__encryption_algorithm = get_encryption_algorithm(passphrase)
        else:
            self.__encryption_algorithm = serialization.NoEncryption()