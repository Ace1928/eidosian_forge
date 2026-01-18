import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def _send_using_smtplib(self):
    """Use python's smtplib to send the email."""
    body = self.body()
    diff = self.get_diff()
    subject = self.subject()
    from_addr = self.from_address()
    to_addrs = self.to()
    header = self.extra_headers()
    msg = EmailMessage(from_addr, to_addrs, subject, body)
    if diff:
        msg.add_inline_attachment(diff, self.diff_filename())
    if header is not None:
        for k, v in header.items():
            msg[k] = v
    smtp = self._smtplib_implementation(self.config)
    smtp.send_email(msg)