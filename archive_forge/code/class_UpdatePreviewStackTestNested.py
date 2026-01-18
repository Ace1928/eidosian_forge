from heat_integrationtests.functional import functional_base
class UpdatePreviewStackTestNested(UpdatePreviewBase):
    template_nested_parent = '\nheat_template_version: 2016-04-08\nresources:\n  nested1:\n    type: nested1.yaml\n'
    template_nested1 = '\nheat_template_version: 2016-04-08\nresources:\n  nested2:\n    type: nested2.yaml\n'
    template_nested2 = '\nheat_template_version: 2016-04-08\nresources:\n  random:\n    type: OS::Heat::RandomString\n'
    template_nested2_2 = '\nheat_template_version: 2016-04-08\nresources:\n  random:\n    type: OS::Heat::RandomString\n  random2:\n    type: OS::Heat::RandomString\n'

    def _get_by_resource_name(self, changes, name, action):
        filtered_l = [x for x in changes[action] if x['resource_name'] == name]
        self.assertEqual(1, len(filtered_l))
        return filtered_l[0]

    def test_nested_resources_nochange(self):
        files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2}
        self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
        result = self.preview_update_stack(self.stack_identifier, template=self.template_nested_parent, files=files, show_nested=True)
        changes = result['resource_changes']
        self.assertEqual(1, len(changes['unchanged']))
        self.assertEqual('random', changes['unchanged'][0]['resource_name'])
        self.assertEqual('nested2', changes['unchanged'][0]['parent_resource'])
        self.assertEqual(2, len(changes['updated']))
        u_nested1 = self._get_by_resource_name(changes, 'nested1', 'updated')
        self.assertNotIn('parent_resource', u_nested1)
        u_nested2 = self._get_by_resource_name(changes, 'nested2', 'updated')
        self.assertEqual('nested1', u_nested2['parent_resource'])
        self.assert_empty_sections(changes, ['replaced', 'deleted', 'added'])

    def test_nested_resources_add(self):
        files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2}
        self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
        files['nested2.yaml'] = self.template_nested2_2
        result = self.preview_update_stack(self.stack_identifier, template=self.template_nested_parent, files=files, show_nested=True)
        changes = result['resource_changes']
        self.assertEqual(1, len(changes['unchanged']))
        self.assertEqual('random', changes['unchanged'][0]['resource_name'])
        self.assertEqual('nested2', changes['unchanged'][0]['parent_resource'])
        self.assertEqual(1, len(changes['added']))
        self.assertEqual('random2', changes['added'][0]['resource_name'])
        self.assertEqual('nested2', changes['added'][0]['parent_resource'])
        self.assert_empty_sections(changes, ['replaced', 'deleted'])

    def test_nested_resources_delete(self):
        files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2_2}
        self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
        files['nested2.yaml'] = self.template_nested2
        result = self.preview_update_stack(self.stack_identifier, template=self.template_nested_parent, files=files, show_nested=True)
        changes = result['resource_changes']
        self.assertEqual(1, len(changes['unchanged']))
        self.assertEqual('random', changes['unchanged'][0]['resource_name'])
        self.assertEqual('nested2', changes['unchanged'][0]['parent_resource'])
        self.assertEqual(1, len(changes['deleted']))
        self.assertEqual('random2', changes['deleted'][0]['resource_name'])
        self.assertEqual('nested2', changes['deleted'][0]['parent_resource'])
        self.assert_empty_sections(changes, ['replaced', 'added'])

    def test_nested_resources_replace(self):
        files = {'nested1.yaml': self.template_nested1, 'nested2.yaml': self.template_nested2}
        self.stack_identifier = self.stack_create(template=self.template_nested_parent, files=files)
        parent_none = self.template_nested_parent.replace('nested1.yaml', 'OS::Heat::None')
        result = self.preview_update_stack(self.stack_identifier, template=parent_none, show_nested=True)
        changes = result['resource_changes']
        self.assertEqual(1, len(changes['replaced']))
        self.assertEqual('nested1', changes['replaced'][0]['resource_name'])
        self.assertEqual(2, len(changes['deleted']))
        d_random = self._get_by_resource_name(changes, 'random', 'deleted')
        self.assertEqual('nested2', d_random['parent_resource'])
        d_nested2 = self._get_by_resource_name(changes, 'nested2', 'deleted')
        self.assertEqual('nested1', d_nested2['parent_resource'])
        self.assert_empty_sections(changes, ['updated', 'unchanged', 'added'])