import base64
import getpass
import os
import signal
import struct
import sys
import tkinter as Tkinter
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox as tkMessageBox
from typing import List, Tuple
from twisted.conch import error
from twisted.conch.client.default import isInKnownHosts
from twisted.conch.ssh import (
from twisted.conch.ui import tkvt100
from twisted.internet import defer, protocol, reactor, tksupport
from twisted.python import log, usage
def _cbVerifyHostKey(self, ans, pubKey, khHost, keyType):
    if ans.lower() not in ('yes', 'no'):
        return deferredAskFrame("Please type  'yes' or 'no': ", 1).addCallback(self._cbVerifyHostKey, pubKey, khHost, keyType)
    if ans.lower() == 'no':
        frame.write('Host key verification failed.\r\n')
        raise error.ConchError('bad host key')
    try:
        frame.write("Warning: Permanently added '%s' (%s) to the list of known hosts.\r\n" % (khHost, {b'ssh-dss': 'DSA', b'ssh-rsa': 'RSA'}[keyType]))
        with open(os.path.expanduser('~/.ssh/known_hosts'), 'a') as known_hosts:
            encodedKey = base64.b64encode(pubKey)
            known_hosts.write(f'\n{khHost} {keyType} {encodedKey}')
    except BaseException:
        log.deferr()
        raise error.ConchError