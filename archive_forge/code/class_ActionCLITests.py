import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class ActionCLITests(base_v2.MistralClientTestBase):
    """Test suite checks commands to work with actions."""

    @classmethod
    def setUpClass(cls):
        super(ActionCLITests, cls).setUpClass()

    def test_action_create_delete(self):
        init_acts = self.mistral_admin('action-create', params=self.act_def)
        self.assertTableStruct(init_acts, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])
        self.assertIn('greeting', [action['Name'] for action in init_acts])
        self.assertIn('farewell', [action['Name'] for action in init_acts])
        action_1 = self.get_item_info(get_from=init_acts, get_by='Name', value='greeting')
        action_2 = self.get_item_info(get_from=init_acts, get_by='Name', value='farewell')
        self.assertEqual('<none>', action_1['Tags'])
        self.assertEqual('<none>', action_2['Tags'])
        self.assertEqual('False', action_1['Is system'])
        self.assertEqual('False', action_2['Is system'])
        self.assertEqual('name', action_1['Input'])
        self.assertEqual('None', action_2['Input'])
        acts = self.mistral_admin('action-list')
        self.assertIn(action_1['Name'], [action['Name'] for action in acts])
        self.assertIn(action_2['Name'], [action['Name'] for action in acts])
        self.mistral_admin('action-delete', params='{0}'.format(action_1['Name']))
        self.mistral_admin('action-delete', params='{0}'.format(action_2['Name']))
        acts = self.mistral_admin('action-list')
        self.assertNotIn(action_1['Name'], [action['Name'] for action in acts])
        self.assertNotIn(action_2['Name'], [action['Name'] for action in acts])

    def test_action_update(self):
        actions = self.action_create(self.act_def)
        created_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
        actions = self.mistral_admin('action-update', params=self.act_def)
        updated_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
        self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
        self.assertEqual(created_action['Name'], updated_action['Name'])
        self.assertEqual(created_action['Updated at'], updated_action['Updated at'])
        actions = self.mistral_admin('action-update', params=self.act_tag_def)
        updated_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
        self.assertEqual('tag, tag1', updated_action['Tags'])
        self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
        self.assertEqual(created_action['Name'], updated_action['Name'])
        self.assertNotEqual(created_action['Updated at'], updated_action['Updated at'])

    def test_action_update_with_id(self):
        acts = self.action_create(self.act_def)
        created_action = self.get_item_info(get_from=acts, get_by='Name', value='greeting')
        action_id = created_action['ID']
        params = '{0} --id {1}'.format(self.act_tag_def, action_id)
        acts = self.mistral_admin('action-update', params=params)
        updated_action = self.get_item_info(get_from=acts, get_by='ID', value=action_id)
        self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
        self.assertEqual(created_action['Name'], updated_action['Name'])
        self.assertNotEqual(created_action['Updated at'], updated_action['Updated at'])

    def test_action_update_truncate_input(self):
        input_value = 'very_long_input_parameter_name_that_should_be_truncated'
        act_def = '\n        version: "2.0"\n        action1:\n          input:\n            - {0}\n          base: std.noop\n        '.format(input_value)
        self.create_file('action.yaml', act_def)
        self.action_create('action.yaml')
        updated_act = self.mistral_admin('action-update', params='action.yaml')
        updated_act_info = self.get_item_info(get_from=updated_act, get_by='Name', value='action1')
        self.assertEqual(updated_act_info['Input'][:-3], input_value[:25])

    def test_action_get_definition(self):
        self.action_create(self.act_def)
        definition = self.mistral_admin('action-get-definition', params='greeting')
        self.assertNotIn('404 Not Found', definition)

    def test_action_get_definition_with_namespace(self):
        self.action_create(self.act_def)
        definition = self.mistral_admin('action-get-definition', params='greeting --namespace test_namespace')
        self.assertNotIn('404 Not Found', definition)

    def test_action_get_with_name(self):
        created = self.action_create(self.act_def)
        action_name = created[0]['Name']
        fetched = self.mistral_admin('action-get', params=action_name)
        fetched_action_name = self.get_field_value(fetched, 'Name')
        self.assertEqual(action_name, fetched_action_name)

    def test_action_list_with_filter(self):
        actions = self.parser.listing(self.mistral('action-list'))
        self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])
        unfiltered_len = len(actions)
        self.assertGreater(unfiltered_len, 0)
        actions = self.parser.listing(self.mistral('action-list', params='--filter name=in:std.echo,std.noop'))
        self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])
        self.assertGreater(unfiltered_len, len(actions))
        action_names = [a['Name'] for a in actions]
        self.assertIn('std.echo', action_names)
        self.assertIn('std.noop', action_names)
        self.assertNotIn('std.ssh', action_names)