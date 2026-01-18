import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
class TestImportZeroMarker(TestCaseForGenericProcessor):

    def test_tag(self):
        handler, branch = self.get_handler()

        def command_list():
            committer = [b'', b'elmer@a.com', time.time(), time.timezone]
            yield commands.TagCommand(b'tag1', b':0', committer, b'tag 1')
        handler.process(command_list)