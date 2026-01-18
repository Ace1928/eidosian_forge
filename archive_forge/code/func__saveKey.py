from __future__ import annotations
import getpass
import os
import platform
import socket
import sys
from collections.abc import Callable
from functools import wraps
from importlib import reload
from typing import Any, Dict, Optional
from twisted.conch.ssh import keys
from twisted.python import failure, filepath, log, usage
def _saveKey(key: keys.Key, options: Dict[Any, Any], inputCollector: Optional[Callable[[str], str]]=None) -> None:
    """
    Persist a SSH key on local filesystem.

    @param key: Key which is persisted on local filesystem.

    @param options:

    @param inputCollector: Dependency injection for testing.
    """
    if inputCollector is None:
        inputCollector = input
    KeyTypeMapping = {'EC': 'ecdsa', 'Ed25519': 'ed25519', 'RSA': 'rsa', 'DSA': 'dsa'}
    keyTypeName = KeyTypeMapping[key.type()]
    filename = options['filename']
    if not filename:
        defaultPath = _getKeyOrDefault(options, inputCollector, keyTypeName)
        newPath = _inputSaveFile(f'Enter file in which to save the key ({defaultPath}): ')
        filename = newPath.strip() or defaultPath
    if os.path.exists(filename):
        print(f'{filename} already exists.')
        yn = inputCollector('Overwrite (y/n)? ')
        if yn[0].lower() != 'y':
            sys.exit()
    if options.get('no-passphrase'):
        options['pass'] = b''
    elif not options['pass']:
        while 1:
            p1 = getpass.getpass('Enter passphrase (empty for no passphrase): ')
            p2 = getpass.getpass('Enter same passphrase again: ')
            if p1 == p2:
                break
            print('Passphrases do not match.  Try again.')
        options['pass'] = p1
    if options.get('private-key-subtype') is None:
        options['private-key-subtype'] = _defaultPrivateKeySubtype(key.type())
    comment = f'{getpass.getuser()}@{socket.gethostname()}'
    fp = filepath.FilePath(filename)
    fp.setContent(key.toString('openssh', subtype=options['private-key-subtype'], passphrase=options['pass']))
    fp.chmod(33152)
    filepath.FilePath(filename + '.pub').setContent(key.public().toString('openssh', comment=comment))
    options = enumrepresentation(options)
    print(f'Your identification has been saved in {filename}')
    print(f'Your public key has been saved in {filename}.pub')
    print(f'The key fingerprint in {options['format']} is:')
    print(key.fingerprint(options['format']))