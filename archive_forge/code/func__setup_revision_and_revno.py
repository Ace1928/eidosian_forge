import subprocess
import tempfile
from ... import errors
from ... import revision as _mod_revision
from ...config import ListOption, Option, bool_from_store, int_from_store
from ...email_message import EmailMessage
from ...smtp_connection import SMTPConnection
def _setup_revision_and_revno(self):
    self.revision = self.repository.get_revision(self._revision_id)
    self.revno = self.branch.revision_id_to_revno(self._revision_id)