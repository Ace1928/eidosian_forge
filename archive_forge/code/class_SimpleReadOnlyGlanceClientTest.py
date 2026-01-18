import re
from tempest.lib import exceptions
from glanceclient.tests.functional import base
class SimpleReadOnlyGlanceClientTest(base.ClientTestBase):
    """Read only functional python-glanceclient tests.

    This only exercises client commands that are read only.
    """

    def test_list_v2(self):
        out = self.glance('--os-image-api-version 2 image-list')
        endpoints = self.parser.listing(out)
        self.assertTableStruct(endpoints, ['ID', 'Name'])

    def test_fake_action(self):
        self.assertRaises(exceptions.CommandFailed, self.glance, 'this-does-not-exist')

    def test_member_list_v2(self):
        try:
            self.glance('--os-image-api-version 2 image-create --name temp')
        except Exception:
            pass
        out = self.glance('--os-image-api-version 2 image-list --visibility private')
        image_list = self.parser.listing(out)
        if len(image_list) > 0:
            param_image_id = '--image-id %s' % image_list[0]['ID']
            out = self.glance('--os-image-api-version 2 member-list', params=param_image_id)
            endpoints = self.parser.listing(out)
            self.assertTableStruct(endpoints, ['Image ID', 'Member ID', 'Status'])
        else:
            param_image_id = '--image-id fake_image_id'
            self.assertRaises(exceptions.CommandFailed, self.glance, '--os-image-api-version 2 member-list', params=param_image_id)

    def test_help(self):
        help_text = self.glance('--os-image-api-version 2 help')
        lines = help_text.split('\n')
        self.assertFirstLineStartsWith(lines, 'usage: glance')
        commands = []
        cmds_start = lines.index('Positional arguments:')
        try:
            cmds_end = lines.index('Options:')
        except ValueError:
            cmds_end = lines.index('Optional arguments:')
        command_pattern = re.compile('^ {4}([a-z0-9\\-\\_]+)')
        for line in lines[cmds_start:cmds_end]:
            match = command_pattern.match(line)
            if match:
                commands.append(match.group(1))
        commands = set(commands)
        wanted_commands = {'bash-completion', 'help', 'image-create', 'image-deactivate', 'image-delete', 'image-download', 'image-list', 'image-reactivate', 'image-show', 'image-tag-delete', 'image-tag-update', 'image-update', 'image-upload', 'location-add', 'location-delete', 'location-update', 'member-create', 'member-delete', 'member-list', 'member-update', 'task-create', 'task-list', 'task-show'}
        self.assertFalse(wanted_commands - commands)

    def test_version(self):
        self.glance('', flags='--version')

    def test_debug_list(self):
        self.glance('--os-image-api-version 2 image-list', flags='--debug')