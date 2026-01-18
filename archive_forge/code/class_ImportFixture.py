import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class ImportFixture(BaseFixture):
    """See DesignateCLI.zone_import_create for __init__ args"""

    def __init__(self, zone_file_contents, user='default', *args, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.zone_file_contents = zone_file_contents

    def _setUp(self):
        super()._setUp()
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.zone_file_contents)
            f.flush()
            self.zone_import = self.client.zone_import_create(*self.args, zone_file_path=f.name, **self.kwargs)
        self.addCleanup(self.cleanup_zone_import, self.client, self.zone_import.id)
        self.addCleanup(ZoneFixture.cleanup_zone, self.client, self.zone_import.zone_id)

    @classmethod
    def cleanup_zone_import(cls, client, zone_import_id):
        try:
            client.zone_import_delete(zone_import_id)
        except CommandFailed:
            pass