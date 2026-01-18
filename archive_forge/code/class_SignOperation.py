from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
class SignOperation(CryptoOperation):
    """Signs a file."""
    keyname = 'private'
    usage = 'usage: %%prog [options] private_key hash_method'
    description = 'Signs a file, outputs the signature. Choose the hash method from %s' % ', '.join(HASH_METHODS)
    operation = 'sign'
    operation_past = 'signature'
    operation_progressive = 'Signing'
    key_class = rsa.PrivateKey
    expected_cli_args = 2
    output_help = 'Name of the file to write the signature to. Written to stdout if this option is not present.'

    def perform_operation(self, indata, priv_key, cli_args):
        """Signs files."""
        hash_method = cli_args[1]
        if hash_method not in HASH_METHODS:
            raise SystemExit('Invalid hash method, choose one of %s' % ', '.join(HASH_METHODS))
        return rsa.sign(indata, priv_key, hash_method)