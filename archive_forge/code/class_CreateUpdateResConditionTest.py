from heat_integrationtests.functional import functional_base
class CreateUpdateResConditionTest(functional_base.FunctionalTestsBase):

    def res_assert_for_prod(self, stack_identifier, bj_prod=True, fj_zone=False, shannxi_provice=False):

        def is_not_deleted(r):
            return r.resource_status != 'DELETE_COMPLETE'
        resources = self.list_resources(stack_identifier, is_not_deleted)
        res_names = set(resources)
        if bj_prod:
            self.assertEqual(4, len(resources))
            self.assertIn('beijing_prod_res', res_names)
            self.assertIn('not_shannxi_res', res_names)
        elif fj_zone:
            self.assertEqual(5, len(resources))
            self.assertIn('fujian_res', res_names)
            self.assertNotIn('beijing_prod_res', res_names)
            self.assertIn('not_shannxi_res', res_names)
            self.assertIn('fujian_prod_res', res_names)
        elif shannxi_provice:
            self.assertEqual(3, len(resources))
            self.assertIn('shannxi_res', res_names)
        else:
            self.assertEqual(3, len(resources))
            self.assertIn('not_shannxi_res', res_names)
        self.assertIn('prod_res', res_names)
        self.assertIn('test_res', res_names)

    def res_assert_for_test(self, stack_identifier, fj_zone=False, shannxi_provice=False):

        def is_not_deleted(r):
            return r.resource_status != 'DELETE_COMPLETE'
        resources = self.list_resources(stack_identifier, is_not_deleted)
        res_names = set(resources)
        if fj_zone:
            self.assertEqual(4, len(resources))
            self.assertIn('fujian_res', res_names)
            self.assertIn('not_shannxi_res', res_names)
        elif shannxi_provice:
            self.assertEqual(3, len(resources))
            self.assertNotIn('fujian_res', res_names)
            self.assertIn('shannxi_res', res_names)
        else:
            self.assertEqual(3, len(resources))
            self.assertIn('not_shannxi_res', res_names)
        self.assertIn('test_res', res_names)
        self.assertIn('test_res1', res_names)
        self.assertNotIn('prod_res', res_names)

    def output_assert_for_prod(self, stack_id, bj_prod=True):
        output = self.client.stacks.output_show(stack_id, 'res_value')['output']
        self.assertEqual('prod_res', output['output_value'])
        test_res_value = self.client.stacks.output_show(stack_id, 'test_res_value')['output']
        self.assertEqual('env_is_prod', test_res_value['output_value'])
        prod_resource = self.client.stacks.output_show(stack_id, 'prod_resource')['output']
        self.assertNotEqual('no_prod_res', prod_resource['output_value'])
        test_res_output = self.client.stacks.output_show(stack_id, 'test_res1_value')['output']
        self.assertEqual('no_test_res1', test_res_output['output_value'])
        beijing_prod_res = self.client.stacks.output_show(stack_id, 'beijing_prod_res')['output']
        if bj_prod:
            self.assertNotEqual('no_prod_res', beijing_prod_res['output_value'])
        else:
            self.assertEqual('no_prod_res', beijing_prod_res['output_value'])

    def output_assert_for_test(self, stack_id):
        output = self.client.stacks.output_show(stack_id, 'res_value')['output']
        self.assertIsNone(output['output_value'])
        test_res_value = self.client.stacks.output_show(stack_id, 'test_res_value')['output']
        self.assertEqual('env_is_test', test_res_value['output_value'])
        prod_resource = self.client.stacks.output_show(stack_id, 'prod_resource')['output']
        self.assertEqual('no_prod_res', prod_resource['output_value'])
        test_res_output = self.client.stacks.output_show(stack_id, 'test_res1_value')['output']
        self.assertEqual('just in test env', test_res_output['output_value'])
        beijing_prod_res = self.client.stacks.output_show(stack_id, 'beijing_prod_res')['output']
        self.assertEqual('no_prod_res', beijing_prod_res['output_value'])

    def test_stack_create_update_cfn_template_test_to_prod(self):
        stack_identifier = self.stack_create(template=cfn_template)
        self.res_assert_for_test(stack_identifier)
        self.output_assert_for_test(stack_identifier)
        parms = {'zone': 'fuzhou'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, fj_zone=True)
        self.output_assert_for_test(stack_identifier)
        parms = {'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, shannxi_provice=True)
        self.output_assert_for_test(stack_identifier)
        parms = {'env_type': 'prod'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier)
        self.output_assert_for_prod(stack_identifier)
        parms = {'env_type': 'prod', 'zone': 'shanghai'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, False)
        self.output_assert_for_prod(stack_identifier, False)
        parms = {'env_type': 'prod', 'zone': 'xiamen'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=True)
        self.output_assert_for_prod(stack_identifier, False)
        parms = {'env_type': 'prod', 'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=True)
        self.output_assert_for_prod(stack_identifier, False)

    def test_stack_create_update_cfn_template_prod_to_test(self):
        parms = {'env_type': 'prod'}
        stack_identifier = self.stack_create(template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier)
        self.output_assert_for_prod(stack_identifier)
        parms = {'zone': 'xiamen', 'env_type': 'prod'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=True)
        self.output_assert_for_prod(stack_identifier, bj_prod=False)
        parms = {'zone': 'xianyang', 'env_type': 'prod'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=True)
        self.output_assert_for_prod(stack_identifier, bj_prod=False)
        parms = {'zone': 'shanghai', 'env_type': 'prod'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, bj_prod=False, fj_zone=False, shannxi_provice=False)
        self.output_assert_for_prod(stack_identifier, bj_prod=False)
        parms = {'env_type': 'test'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_test(stack_identifier)
        self.output_assert_for_test(stack_identifier)
        parms = {'env_type': 'test', 'zone': 'fuzhou'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, fj_zone=True)
        self.output_assert_for_test(stack_identifier)
        parms = {'env_type': 'test', 'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=cfn_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, fj_zone=False, shannxi_provice=True)
        self.output_assert_for_test(stack_identifier)

    def test_stack_create_update_hot_template_test_to_prod(self):
        stack_identifier = self.stack_create(template=hot_template)
        self.res_assert_for_test(stack_identifier)
        self.output_assert_for_test(stack_identifier)
        parms = {'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, shannxi_provice=True)
        self.output_assert_for_test(stack_identifier)
        parms = {'env_type': 'prod'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier)
        self.output_assert_for_prod(stack_identifier)
        parms = {'env_type': 'prod', 'zone': 'shanghai'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, False)
        self.output_assert_for_prod(stack_identifier, False)
        parms = {'env_type': 'prod', 'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, False, shannxi_provice=True)
        self.output_assert_for_prod(stack_identifier, False)

    def test_stack_create_update_hot_template_prod_to_test(self):
        parms = {'env_type': 'prod'}
        stack_identifier = self.stack_create(template=hot_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier)
        self.output_assert_for_prod(stack_identifier)
        parms = {'env_type': 'prod', 'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_prod(stack_identifier, False, shannxi_provice=True)
        self.output_assert_for_prod(stack_identifier, False)
        parms = {'env_type': 'test'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_test(stack_identifier)
        self.output_assert_for_test(stack_identifier)
        parms = {'env_type': 'test', 'zone': 'xianyang'}
        self.update_stack(stack_identifier, template=hot_template, parameters=parms)
        self.res_assert_for_test(stack_identifier, fj_zone=False, shannxi_provice=True)
        self.output_assert_for_test(stack_identifier)

    def test_condition_rename(self):
        stack_identifier = self.stack_create(template=before_rename_tmpl)
        self.update_stack(stack_identifier, template=after_rename_tmpl)
        self.update_stack(stack_identifier, template=fail_rename_tmpl, expected_status='UPDATE_FAILED')
        self.update_stack(stack_identifier, template=recover_rename_tmpl)