import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class ExportFixture(BaseFixture):
    """See DesignateCLI.zone_export_create for __init__ args"""

    def __init__(self, zone, user='default', *args, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.zone = zone

    def _setUp(self):
        super()._setUp()
        self.zone_export = self.client.zone_export_create(*self.args, zone_id=self.zone.id, **self.kwargs)
        self.addCleanup(self.cleanup_zone_export, self.client, self.zone_export.id)
        self.addCleanup(ZoneFixture.cleanup_zone, self.client, self.zone.id)

    @classmethod
    def cleanup_zone_export(cls, client, zone_export_id):
        try:
            client.zone_export_delete(zone_export_id)
        except CommandFailed:
            pass