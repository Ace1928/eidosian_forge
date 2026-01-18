import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class BlacklistFixture(BaseFixture):
    """See DesignateCLI.zone_blacklist_create for __init__ args"""

    def __init__(self, user='admin', *args, **kwargs):
        super().__init__(*args, user=user, **kwargs)

    def _setUp(self):
        super()._setUp()
        self.blacklist = self.client.zone_blacklist_create(*self.args, **self.kwargs)
        self.addCleanup(self.cleanup_blacklist, self.client, self.blacklist.id)

    @classmethod
    def cleanup_blacklist(cls, client, blacklist_id):
        try:
            client.zone_blacklist_delete(blacklist_id)
        except CommandFailed:
            pass