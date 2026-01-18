import re
from unittest import mock
from testtools import matchers
from magnumclient.tests import utils
class TestCommandLineArgument(utils.TestCase):
    _unrecognized_arg_error = ['.*?^usage: ', '.*?^error: unrecognized arguments:', ".*?^Try 'magnum help ' for more information."]
    _mandatory_group_arg_error = ['.*?^usage: ', '.*?^error: one of the arguments', ".*?^Try 'magnum help "]
    _too_many_group_arg_error = ['.*?^usage: ', '.*?^error: (argument \\-\\-[a-z\\-]*: not allowed with argument )', ".*?^Try 'magnum help "]
    _mandatory_arg_error = ['.*?^usage: ', '.*?^error: (the following arguments|argument)', ".*?^Try 'magnum help "]
    _duplicate_arg_error = ['.*?^usage: ', '.*?^error: (Duplicate "<.*>" arguments:)', ".*?^Try 'magnum help "]
    _deprecated_warning = ['.*(WARNING: The \\-\\-[a-z\\-]* parameter is deprecated)+', '.*(Use the [\\<\\-a-z\\-\\>]* (positional )*parameter to avoid seeing this message)+']
    _few_argument_error = ['.*?^usage: magnum ', '.*?^error: (the following arguments|too few arguments)', ".*?^Try 'magnum help "]
    _invalid_value_error = ['.*?^usage: ', '.*?^error: argument .*: invalid .* value:', ".*?^Try 'magnum help "]

    def setUp(self):
        super(TestCommandLineArgument, self).setUp()
        self.make_env(fake_env=FAKE_ENV)
        session_client = mock.patch('magnumclient.common.httpclient.SessionClient')
        session_client.start()
        loader = mock.patch('keystoneauth1.loading.get_plugin_loader')
        loader.start()
        session = mock.patch('keystoneauth1.session.Session')
        session.start()
        self.addCleanup(session_client.stop)
        self.addCleanup(loader.stop)
        self.addCleanup(session.stop)

    def _test_arg_success(self, command, keyword=None):
        stdout, stderr = self.shell(command)
        if keyword:
            self.assertIn(keyword, stdout + stderr)

    def _test_arg_failure(self, command, error_msg):
        stdout, stderr = self.shell(command, (2,))
        for line in error_msg:
            self.assertThat(stdout + stderr, matchers.MatchesRegex(line, re.DOTALL | re.MULTILINE))