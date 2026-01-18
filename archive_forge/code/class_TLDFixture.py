import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class TLDFixture(BaseFixture):
    """See DesignateCLI.tld_create for __init__ args"""

    def __init__(self, user='admin', *args, **kwargs):
        super().__init__(*args, user=user, **kwargs)

    def _setUp(self):
        super()._setUp()
        self.tld = self.client.tld_create(*self.args, **self.kwargs)
        self.addCleanup(self.cleanup_tld, self.client, self.tld.id)

    @classmethod
    def cleanup_tld(cls, client, tld_id):
        try:
            client.tld_delete(tld_id)
        except CommandFailed:
            pass