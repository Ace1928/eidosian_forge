import os
from ctypes import *
def SendMail(recipient, subject='', body='', attachfiles=''):
    """Post an e-mail message using Simple MAPI

    recipient - string: address to send to (multiple addresses separated with a semicolon)
    subject   - string: subject header
    body      - string: message text
    attach    - string: files to attach (multiple attachments separated with a semicolon)
    """
    attach = []
    AttachWork = attachfiles.split(';')
    for f in AttachWork:
        if os.path.exists(f):
            attach.append(os.path.abspath(f))
    restore = os.getcwd()
    try:
        session = _logon()
        try:
            _sendMail(session, recipient, subject, body, attach)
        finally:
            _logoff(session)
    finally:
        os.chdir(restore)