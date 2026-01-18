import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def _send_using_process(self):
    """Spawn a 'mail' subprocess to send the email."""
    with tempfile.NamedTemporaryFile() as msgfile:
        msgfile.write(self.body().encode('utf8'))
        diff = self.get_diff()
        if diff:
            msgfile.write(diff)
        msgfile.flush()
        msgfile.seek(0)
        process = subprocess.Popen(self._command_line(), stdin=msgfile.fileno())
        rc = process.wait()
        if rc != 0:
            raise errors.BzrError('Failed to send email: exit status {}'.format(rc))