import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
class PureProxy(SMTPServer):

    def __init__(self, *args, **kwargs):
        if 'enable_SMTPUTF8' in kwargs and kwargs['enable_SMTPUTF8']:
            raise ValueError('PureProxy does not support SMTPUTF8.')
        super(PureProxy, self).__init__(*args, **kwargs)

    def process_message(self, peer, mailfrom, rcpttos, data):
        lines = data.split('\n')
        i = 0
        for line in lines:
            if not line:
                break
            i += 1
        lines.insert(i, 'X-Peer: %s' % peer[0])
        data = NEWLINE.join(lines)
        refused = self._deliver(mailfrom, rcpttos, data)
        print('we got some refusals:', refused, file=DEBUGSTREAM)

    def _deliver(self, mailfrom, rcpttos, data):
        import smtplib
        refused = {}
        try:
            s = smtplib.SMTP()
            s.connect(self._remoteaddr[0], self._remoteaddr[1])
            try:
                refused = s.sendmail(mailfrom, rcpttos, data)
            finally:
                s.quit()
        except smtplib.SMTPRecipientsRefused as e:
            print('got SMTPRecipientsRefused', file=DEBUGSTREAM)
            refused = e.recipients
        except (OSError, smtplib.SMTPException) as e:
            print('got', e.__class__, file=DEBUGSTREAM)
            errcode = getattr(e, 'smtp_code', -1)
            errmsg = getattr(e, 'smtp_error', 'ignore')
            for r in rcpttos:
                refused[r] = (errcode, errmsg)
        return refused