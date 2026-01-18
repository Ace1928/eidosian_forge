import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
def _get_pkey_object(self, key, password=None):
    """
        Try to detect private key type and return paramiko.PKey object.

        # NOTE: Paramiko only supports key in PKCS#1 PEM format.
        """
    key_types = [(paramiko.RSAKey, 'RSA'), (paramiko.DSSKey, 'DSA'), (paramiko.ECDSAKey, 'EC')]
    paramiko_version = getattr(paramiko, '__version__', '0.0.0')
    paramiko_version = tuple((int(c) for c in paramiko_version.split('.')))
    if paramiko_version >= (2, 2, 0):
        key_types.append((paramiko.ed25519key.Ed25519Key, 'Ed25519'))
    for cls, key_type in key_types:
        key_split = key.strip().splitlines()
        if key_split[0] == '-----BEGIN PRIVATE KEY-----' and key_split[-1] == '-----END PRIVATE KEY-----':
            key_split[0] = '-----BEGIN %s PRIVATE KEY-----' % key_type
            key_split[-1] = '-----END %s PRIVATE KEY-----' % key_type
            key_value = '\n'.join(key_split)
        else:
            key_value = key
        try:
            key = cls.from_private_key(StringIO(key_value), password)
        except paramiko.ssh_exception.PasswordRequiredException as e:
            raise e
        except (paramiko.ssh_exception.SSHException, AssertionError) as e:
            if 'private key file checkints do not match' in str(e).lower():
                msg = 'Invalid password provided for encrypted key. Original error: %s' % str(e)
                raise paramiko.ssh_exception.SSHException(msg)
            pass
        else:
            return key
    msg = 'Invalid or unsupported key type (only RSA, DSS, ECDSA and Ed25519 keys in PEM format are supported). For more information on  supported key file types, see %s' % SUPPORTED_KEY_TYPES_URL
    raise paramiko.ssh_exception.SSHException(msg)