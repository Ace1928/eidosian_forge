import weakref
import threading
import time
import re
from paramiko.common import (
from paramiko.message import Message
from paramiko.util import b, u
from paramiko.ssh_exception import (
from paramiko.server import InteractiveQuery
from paramiko.ssh_gss import GSSAuth, GSS_EXCEPTIONS
def _finalize_pubkey_algorithm(self, key_type):
    if 'rsa' not in key_type:
        return key_type
    self._log(DEBUG, 'Finalizing pubkey algorithm for key of type {!r}'.format(key_type))
    if key_type.endswith('-cert-v01@openssh.com') and re.search('-OpenSSH_(?:[1-6]|7\\.[0-7])', self.transport.remote_version):
        pubkey_algo = 'ssh-rsa-cert-v01@openssh.com'
        self.transport._agreed_pubkey_algorithm = pubkey_algo
        self._log(DEBUG, 'OpenSSH<7.8 + RSA cert = forcing ssh-rsa!')
        self._log(DEBUG, 'Agreed upon {!r} pubkey algorithm'.format(pubkey_algo))
        return pubkey_algo
    my_algos = [x for x in self.transport.preferred_pubkeys if 'rsa' in x]
    self._log(DEBUG, 'Our pubkey algorithm list: {}'.format(my_algos))
    if not my_algos:
        raise SSHException('An RSA key was specified, but no RSA pubkey algorithms are configured!')
    server_algo_str = u(self.transport.server_extensions.get('server-sig-algs', b('')))
    pubkey_algo = None
    if server_algo_str:
        server_algos = server_algo_str.split(',')
        self._log(DEBUG, 'Server-side algorithm list: {}'.format(server_algos))
        agreement = list(filter(server_algos.__contains__, my_algos))
        if agreement:
            pubkey_algo = agreement[0]
            self._log(DEBUG, 'Agreed upon {!r} pubkey algorithm'.format(pubkey_algo))
        else:
            self._log(DEBUG, 'No common pubkey algorithms exist! Dying.')
            err = 'Unable to agree on a pubkey algorithm for signing a {!r} key!'
            raise AuthenticationException(err.format(key_type))
    else:
        pubkey_algo = self._choose_fallback_pubkey_algorithm(key_type, my_algos)
    if key_type.endswith('-cert-v01@openssh.com'):
        pubkey_algo += '-cert-v01@openssh.com'
    self.transport._agreed_pubkey_algorithm = pubkey_algo
    return pubkey_algo