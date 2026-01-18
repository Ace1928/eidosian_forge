import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class RecordsetFixture(BaseFixture):
    """See DesignateCLI.recordset_create for __init__ args"""

    def _setUp(self):
        super()._setUp()
        self.recordset = self.client.recordset_create(*self.args, **self.kwargs)
        self.addCleanup(self.cleanup_recordset, self.client, self.recordset.zone_id, self.recordset.id)

    @classmethod
    def cleanup_recordset(cls, client, zone_id, recordset_id):
        try:
            client.recordset_delete(zone_id, recordset_id)
        except CommandFailed:
            pass