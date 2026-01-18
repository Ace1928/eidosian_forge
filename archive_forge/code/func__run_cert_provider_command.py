import json
import logging
from os import path
import re
import subprocess
import six
from google.auth import exceptions
def _run_cert_provider_command(command, expect_encrypted_key=False):
    """Run the provided command, and return client side mTLS cert, key and
    passphrase.

    Args:
        command (List[str]): cert provider command.
        expect_encrypted_key (bool): If encrypted private key is expected.

    Returns:
        Tuple[bytes, bytes, bytes]: client certificate bytes in PEM format, key
            bytes in PEM format and passphrase bytes.

    Raises:
        google.auth.exceptions.ClientCertError: if problems occurs when running
            the cert provider command or generating cert, key and passphrase.
    """
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
    except OSError as caught_exc:
        new_exc = exceptions.ClientCertError(caught_exc)
        six.raise_from(new_exc, caught_exc)
    if process.returncode != 0:
        raise exceptions.ClientCertError('Cert provider command returns non-zero status code %s' % process.returncode)
    cert_match = re.findall(_CERT_REGEX, stdout)
    if len(cert_match) != 1:
        raise exceptions.ClientCertError('Client SSL certificate is missing or invalid')
    key_match = re.findall(_KEY_REGEX, stdout)
    if len(key_match) != 1:
        raise exceptions.ClientCertError('Client SSL key is missing or invalid')
    passphrase_match = re.findall(_PASSPHRASE_REGEX, stdout)
    if expect_encrypted_key:
        if len(passphrase_match) != 1:
            raise exceptions.ClientCertError('Passphrase is missing or invalid')
        if b'ENCRYPTED' not in key_match[0]:
            raise exceptions.ClientCertError('Encrypted private key is expected')
        return (cert_match[0], key_match[0], passphrase_match[0].strip())
    if b'ENCRYPTED' in key_match[0]:
        raise exceptions.ClientCertError('Encrypted private key is not expected')
    if len(passphrase_match) > 0:
        raise exceptions.ClientCertError('Passphrase is not expected')
    return (cert_match[0], key_match[0], None)