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
class GeneralOptions(usage.Options):
    synopsis = 'Usage:    ckeygen [options]\n '
    longdesc = 'ckeygen manipulates public/private keys in various ways.'
    optParameters = [['bits', 'b', None, 'Number of bits in the key to create.'], ['filename', 'f', None, 'Filename of the key file.'], ['type', 't', None, 'Specify type of key to create.'], ['comment', 'C', None, 'Provide new comment.'], ['newpass', 'N', None, 'Provide new passphrase.'], ['pass', 'P', None, 'Provide old passphrase.'], ['format', 'o', 'sha256-base64', 'Fingerprint format of key file.'], ['private-key-subtype', None, None, 'OpenSSH private key subtype to write ("PEM" or "v1").']]
    optFlags = [['fingerprint', 'l', 'Show fingerprint of key file.'], ['changepass', 'p', 'Change passphrase of private key file.'], ['quiet', 'q', 'Quiet.'], ['no-passphrase', None, 'Create the key with no passphrase.'], ['showpub', 'y', 'Read private key file and print public key.']]
    compData = usage.Completions(optActions={'type': usage.CompleteList(list(supportedKeyTypes.keys())), 'private-key-subtype': usage.CompleteList(['PEM', 'v1'])})