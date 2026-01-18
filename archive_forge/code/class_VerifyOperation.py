from __future__ import with_statement, print_function
import abc
import sys
from optparse import OptionParser
import rsa
import rsa.pkcs1
class VerifyOperation(CryptoOperation):
    """Verify a signature."""
    keyname = 'public'
    usage = 'usage: %%prog [options] public_key signature_file'
    description = 'Verifies a signature, exits with status 0 upon success, prints an error message and exits with status 1 upon error.'
    operation = 'verify'
    operation_past = 'verified'
    operation_progressive = 'Verifying'
    key_class = rsa.PublicKey
    expected_cli_args = 2
    has_output = False

    def perform_operation(self, indata, pub_key, cli_args):
        """Verifies files."""
        signature_file = cli_args[1]
        with open(signature_file, 'rb') as sigfile:
            signature = sigfile.read()
        try:
            rsa.verify(indata, signature, pub_key)
        except rsa.VerificationError:
            raise SystemExit('Verification failed.')
        print('Verification OK', file=sys.stderr)