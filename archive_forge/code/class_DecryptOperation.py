from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
class DecryptOperation(CryptoOperation):
    """Decrypts a file."""
    keyname = 'private'
    description = 'Decrypts a file. The original file must be shorter than the key length in order to have been encrypted.'
    operation = 'decrypt'
    operation_past = 'decrypted'
    operation_progressive = 'decrypting'
    key_class = rsa.PrivateKey

    def perform_operation(self, indata, priv_key, cli_args=None):
        """Decrypts files."""
        return rsa.decrypt(indata, priv_key)