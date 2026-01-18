from gslib.utils import system_util
from gslib.utils.user_agent_helper import GetUserAgent
import gslib.tests.testcase as testcase
import six
from six import add_move, MovedModule
from six.moves import mock
class TestUserAgentHelper(testcase.GsUtilUnitTestCase):
    """Unit tests for the GetUserAgent helper function."""

    @mock.patch('gslib.VERSION', '4_test')
    def testNoArgs(self):
        self.assertRegex(GetUserAgent([]), '^ gsutil/4_test \\([^\\)]+\\)')

    def testAnalyticsFlag(self):
        self.assertRegex(GetUserAgent([], False), 'analytics/enabled')
        self.assertRegex(GetUserAgent([], True), 'analytics/disabled')

    @mock.patch.object(system_util, 'IsRunningInteractively')
    def testInteractiveFlag(self, mock_interactive):
        mock_interactive.return_value = True
        self.assertRegex(GetUserAgent([]), 'interactive/True')
        mock_interactive.return_value = False
        self.assertRegex(GetUserAgent([]), 'interactive/False')

    def testHelp(self):
        self.assertRegex(GetUserAgent(['help']), 'command/help')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCp(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp', '-r', '-Z', '1.txt', 'gs://dst']), 'command/cp$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCpNotEnoughArgs(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp']), 'command/cp$')
        self.assertRegex(GetUserAgent(['cp', '1.txt']), 'command/cp$')
        self.assertRegex(GetUserAgent(['cp', '-r', '1.ts']), 'command/cp$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCpEncoding(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp', 'öne', 'twö']), 'command/cp$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testRsync(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['rsync', '1.txt', 'gs://dst']), 'command/rsync$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testMv(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['mv', 'gs://src/1.txt', 'gs://dst/1.txt']), 'command/mv$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCpCloudToCloud(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp', '-r', 'gs://src', 'gs://dst']), 'command/cp$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCpForcedDaisyChain(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp', '-D', 'gs://src', 'gs://dst']), 'command/cp$')

    def testCpDaisyChain(self):
        self.assertRegex(GetUserAgent(['cp', '-r', '-Z', 'gs://src', 's3://dst']), 'command/cp-DaisyChain')
        self.assertRegex(GetUserAgent(['mv', 'gs://src/1.txt', 's3://dst/1.txt']), 'command/mv-DaisyChain')
        self.assertRegex(GetUserAgent(['rsync', '-r', 'gs://src', 's3://dst']), 'command/rsync-DaisyChain')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testPassOnInvalidUrlError(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['cp', '-r', '-Z', 'bad://src', 's3://dst']), 'command/cp$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testRewriteEncryptionKey(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['rewrite', '-k', 'gs://dst']), 'command/rewrite-k$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testRewriteStorageClass(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['rewrite', '-s', 'gs://dst']), 'command/rewrite-s$')

    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testRewriteEncryptionKeyAndStorageClass(self, mock_invoked):
        mock_invoked.return_value = False
        self.assertRegex(GetUserAgent(['rewrite', '-k', '-s', 'gs://dst']), 'command/rewrite-k-s$')

    @mock.patch.object(system_util, 'CloudSdkVersion')
    @mock.patch.object(system_util, 'InvokedViaCloudSdk')
    def testCloudSdk(self, mock_invoked, mock_version):
        mock_invoked.return_value = True
        mock_version.return_value = '500.1'
        self.assertRegex(GetUserAgent(['help']), 'google-cloud-sdk/500.1$')
        mock_invoked.return_value = False
        mock_version.return_value = '500.1'
        self.assertRegex(GetUserAgent(['help']), 'command/help$')