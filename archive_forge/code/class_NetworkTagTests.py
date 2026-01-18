import uuid
from openstackclient.tests.functional import base
class NetworkTagTests(NetworkTests):
    """Functional tests with tag operation"""
    base_command = None

    def test_tag_operation(self):
        cmd_output = self.openstack('token issue ', parse_output=True)
        auth_project_id = cmd_output['project_id']
        name1 = self._create_resource_and_tag_check('', [])
        name2 = self._create_resource_and_tag_check('--tag red --tag blue', ['red', 'blue'])
        name3 = self._create_resource_and_tag_check('--no-tag', [])
        self._set_resource_and_tag_check('set', name1, '--tag red --tag green', ['red', 'green'])
        list_expected = ((name1, ['red', 'green']), (name2, ['red', 'blue']), (name3, []))
        self._list_tag_check(auth_project_id, list_expected)
        self._set_resource_and_tag_check('set', name1, '--tag blue', ['red', 'green', 'blue'])
        self._set_resource_and_tag_check('set', name1, '--no-tag --tag yellow --tag orange --tag purple', ['yellow', 'orange', 'purple'])
        self._set_resource_and_tag_check('unset', name1, '--tag yellow', ['orange', 'purple'])
        self._set_resource_and_tag_check('unset', name1, '--all-tag', [])
        self._set_resource_and_tag_check('set', name2, '--no-tag', [])

    def _list_tag_check(self, project_id, expected):
        cmd_output = self.openstack('{} list --long --project {}'.format(self.base_command, project_id), parse_output=True)
        for name, tags in expected:
            net = [n for n in cmd_output if n['Name'] == name][0]
            self.assertEqual(set(tags), set(net['Tags']))

    def _create_resource_for_tag_test(self, name, args):
        return self.openstack('{} create {} {}'.format(self.base_command, args, name), parse_output=True)

    def _create_resource_and_tag_check(self, args, expected):
        name = uuid.uuid4().hex
        cmd_output = self._create_resource_for_tag_test(name, args)
        self.addCleanup(self.openstack, '{} delete {}'.format(self.base_command, name))
        self.assertIsNotNone(cmd_output['id'])
        self.assertEqual(set(expected), set(cmd_output['tags']))
        return name

    def _set_resource_and_tag_check(self, command, name, args, expected):
        cmd_output = self.openstack('{} {} {} {}'.format(self.base_command, command, args, name))
        self.assertFalse(cmd_output)
        cmd_output = self.openstack('{} show {}'.format(self.base_command, name), parse_output=True)
        self.assertEqual(set(expected), set(cmd_output['tags']))