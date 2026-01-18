import re
import ddt
from manilaclient.tests.functional import base
@ddt.ddt
class ManilaClientTestCommonReadOnly(base.BaseTestCase):

    @ddt.data('admin', 'user')
    def test_manila_version(self, role):
        self.clients[role].manila('', flags='--version')

    @ddt.data('admin', 'user')
    def test_help(self, role):
        help_text = self.clients[role].manila('help')
        lines = help_text.split('\n')
        self.assertFirstLineStartsWith(lines, 'usage: manila')
        commands = []
        cmds_start = lines.index('Positional arguments:')
        try:
            cmds_end = lines.index('Optional arguments:')
        except ValueError:
            cmds_end = lines.index('Options:')
        command_pattern = re.compile('^ {4}([a-z0-9\\-\\_]+)')
        for line in lines[cmds_start:cmds_end]:
            match = command_pattern.match(line)
            if match:
                commands.append(match.group(1))
        commands = set(commands)
        wanted_commands = set(('absolute-limits', 'list', 'help', 'quota-show', 'access-list', 'snapshot-list', 'access-allow', 'access-deny', 'share-network-list', 'security-service-list'))
        self.assertFalse(wanted_commands - commands)

    @ddt.data('admin', 'user')
    def test_credentials(self, role):
        self.clients[role].manila('credentials')

    @ddt.data('admin', 'user')
    def test_list_extensions(self, role):
        roles = self.parser.listing(self.clients[role].manila('list-extensions'))
        self.assertTableStruct(roles, ['Name', 'Summary', 'Alias', 'Updated'])

    @ddt.data('admin', 'user')
    def test_endpoints(self, role):
        self.clients[role].manila('endpoints')