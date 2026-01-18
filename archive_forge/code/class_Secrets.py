from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
class Secrets(Logger):
    """Logger for secrets."""

    def _Print(self, action, secret_ref):
        self.Print('{action} secret [{secret}].'.format(action=action, secret=secret_ref.Name()))

    def Created(self, secret_ref):
        self._Print('Created', secret_ref)

    def Deleted(self, secret_ref):
        self._Print('Deleted', secret_ref)

    def Updated(self, secret_ref):
        self._Print('Updated', secret_ref)

    def UpdatedReplication(self, secret_ref):
        self._Print('Updated replication for', secret_ref)