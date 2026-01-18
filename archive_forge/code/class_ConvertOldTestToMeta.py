import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
class ConvertOldTestToMeta(controldir.Converter):
    """A trivial converter, used for testing."""

    def convert(self, to_convert, pb):
        ui.ui_factory.note('starting upgrade from old test format to 2a')
        to_convert.control_transport.put_bytes('branch-format', bzrdir.BzrDirMetaFormat1().get_format_string(), mode=to_convert._get_file_mode())
        return controldir.ControlDir.open(to_convert.user_url)