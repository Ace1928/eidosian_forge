import re
import socket
import collections
import datetime
import sys
import warnings
from email.header import decode_header as _email_decode_header
from socket import _GLOBAL_DEFAULT_TIMEOUT
def _setreadermode(self):
    try:
        self.welcome = self._shortcmd('mode reader')
    except NNTPPermanentError:
        pass
    except NNTPTemporaryError as e:
        if e.response.startswith('480'):
            self.readermode_afterauth = True
        else:
            raise