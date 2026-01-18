import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
class SFTPTests(TestCaseWithSFTPServer):
    """Tests for upgrade over sftp."""

    def test_upgrade_url(self):
        self.run_bzr('init --format=pack-0.92')
        t = self.get_transport()
        url = t.base
        display_url = urlutils.unescape_for_display(url, 'utf-8')
        out, err = self.run_bzr(['upgrade', '--format=2a', url])
        backup_dir = 'backup.bzr.~1~'
        self.assertEqualDiff('Upgrading branch {} ...\nstarting upgrade of {}\nmaking backup of {}.bzr\n  to {}{}\nstarting repository conversion\nrepository converted\nfinished\n'.format(display_url, display_url, display_url, display_url, backup_dir), out)
        self.assertEqual('', err)