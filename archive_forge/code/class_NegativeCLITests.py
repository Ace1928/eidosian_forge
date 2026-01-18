import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class NegativeCLITests(base_v2.MistralClientTestBase):
    """This class contains negative tests."""

    def test_wb_list_extra_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-list', params='param')

    def test_wb_get_unexist_wb(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-get', params='wb')

    def test_wb_get_without_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-get')

    def test_wb_create_same_name(self):
        self.workbook_create(self.wb_def)
        self.assertRaises(exceptions.CommandFailed, self.workbook_create, self.wb_def)

    def test_wb_create_with_wrong_path_to_definition(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook_create', 'wb')

    def test_wb_delete_unexist_wb(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-delete', params='wb')

    def test_wb_update_wrong_path_to_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update', params='wb')

    def test_wb_update_nonexistant_wb(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update', params=self.wb_with_tags_def)

    def test_wb_create_empty_def(self):
        self.create_file('empty')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-create', params='empty')

    def test_wb_update_empty_def(self):
        self.create_file('empty')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update', params='empty')

    def test_wb_get_definition_unexist_wb(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-get-definition', params='wb')

    def test_wb_create_invalid_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-create', params=self.wf_def)

    def test_wb_update_invalid_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update', params=self.wf_def)

    def test_wb_update_without_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workbook-update')

    def test_wf_list_extra_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-list', params='param')

    def test_wf_get_unexist_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-get', params='wf')

    def test_wf_get_without_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-get')

    def test_wf_create_without_definition(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='')

    def test_wf_create_with_wrong_path_to_definition(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='wf')

    def test_wf_delete_unexist_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-delete', params='wf')

    def test_wf_update_unexist_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-update', params='wf')

    def test_wf_get_definition_unexist_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-get-definition', params='wf')

    def test_wf_get_definition_missed_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-get-definition')

    def test_wf_create_invalid_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params=self.wb_def)

    def test_wf_update_invalid_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-update', params=self.wb_def)

    def test_wf_create_empty_def(self):
        self.create_file('empty')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-create', params='empty')

    def test_wf_update_empty_def(self):
        self.create_file('empty')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-update', params='empty')

    def test_ex_list_extra_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-list', params='param')

    def test_ex_create_unexist_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create', params='wf')

    def test_ex_create_unexist_task(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create', params='%s param {}' % wf[0]['Name'])

    def test_ex_create_with_invalid_input(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create', params='%s input' % wf[0]['Name'])

    def test_ex_get_nonexist_execution(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-get', params='%s id' % wf[0]['Name'])

    def test_ex_create_without_wf_name(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create')

    def test_ex_create_reverse_wf_without_start_task(self):
        wf = self.workflow_create(self.wf_def)
        self.create_file('input', '{\n    "farewell": "Bye"\n}\n')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create ', params=wf[1]['Name'])

    def test_ex_create_missed_input(self):
        self.create_file('empty')
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create empty', params=wf[1]['Name'])

    def test_ex_update_both_state_and_description(self):
        wf = self.workflow_create(self.wf_def)
        execution = self.execution_create(params=wf[0]['Name'])
        exec_id = self.get_field_value(execution, 'ID')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-update', params='%s -s ERROR -d update' % exec_id)

    def test_ex_delete_nonexistent_execution(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-delete', params='1a2b3c')

    def test_tr_create_without_pattern(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr %s {}' % wf[0]['Name'])

    def test_tr_create_invalid_pattern(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr %s {} --pattern "q"' % wf[0]['Name'])

    def test_tr_create_invalid_pattern_value_out_of_range(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr %s {} --pattern "80 * * * *"' % wf[0]['Name'])

    def test_tr_create_nonexistent_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --pattern "* * * * *"')

    def test_tr_delete_nonexistant_tr(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-delete', params='tr')

    def test_tr_get_nonexistant_tr(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-get', params='tr')

    def test_tr_create_invalid_count(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --pattern "* * * * *" --count q')

    def test_tr_create_negative_count(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --pattern "* * * * *" --count -1')

    def test_tr_create_invalid_first_date(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --pattern "* * * * *" --first-date "q"')

    def test_tr_create_count_only(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --count 42')

    def test_tr_create_date_and_count_without_pattern(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'cron-trigger-create', params='tr wb.wf1 {} --count 42 --first-time "4242-12-25 13:37"')

    def test_event_tr_create_missing_argument(self):
        wf = self.workflow_create(self.wf_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-create', params='tr %s exchange topic' % wf[0]['ID'])

    def test_event_tr_create_nonexistent_wf(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-create', params='456 4307362e-4a4a-4021-aa58-0fab23c9c751 exchange topic event {} ')

    def test_event_tr_delete_nonexistent_tr(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-delete', params='789')

    def test_event_tr_get_nonexistent_tr(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-get', params='789')

    def test_action_get_nonexistent(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral, 'action-get', params='nonexist')

    def test_action_double_creation(self):
        self.action_create(self.act_def)
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='{0}'.format(self.act_def))

    def test_action_create_without_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='')

    def test_action_create_invalid_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='{0}'.format(self.wb_def))

    def test_action_delete_nonexistent_act(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-delete', params='nonexist')

    def test_action_delete_standard_action(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-delete', params='heat.events_get')

    def test_action_get_definition_nonexistent_action(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-get-definition', params='nonexist')

    def test_task_get_nonexistent_task(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'task-get', params='nonexist')

    def test_env_get_without_param(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-get')

    def test_env_get_nonexistent(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-get', params='nonexist')

    def test_env_create_same_name(self):
        self.create_file('env.yaml', 'name: env\ndescription: Test env\nvariables:\n  var: "value"')
        self.environment_create('env.yaml')
        self.assertRaises(exceptions.CommandFailed, self.environment_create, 'env.yaml')

    def test_env_create_empty(self):
        self.create_file('env.yaml')
        self.assertRaises(exceptions.CommandFailed, self.environment_create, 'env.yaml')

    def test_env_create_with_wrong_path_to_definition(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution_create', 'env')

    def test_env_delete_unexist_env(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-delete', params='env')

    def test_env_update_wrong_path_to_def(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-update', params='env')

    def test_env_update_empty(self):
        self.create_file('env.yaml')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-update', params='env')

    def test_env_update_nonexistant_env(self):
        self.create_file('env.yaml', 'name: envvariables:\n  var: "value"')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-update', params='env.yaml')

    def test_env_create_without_name(self):
        self.create_file('env.yaml', 'variables:\n  var: "value"')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-create', params='env.yaml')

    def test_env_create_without_variables(self):
        self.create_file('env.yaml', 'name: env')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'environment-create', params='env.yaml')

    def test_action_execution_get_without_params(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-execution-get')

    def test_action_execution_get_unexistent_obj(self):
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-execution-get', params='123456')

    def test_action_execution_update(self):
        wfs = self.workflow_create(self.wf_def)
        direct_wf_exec = self.execution_create(wfs[0]['Name'])
        direct_ex_id = self.get_field_value(direct_wf_exec, 'ID')
        self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-execution-update', params='%s ERROR' % direct_ex_id)

    def test_target_action_execution(self):
        command = '--debug --os-target-tenant-name={tenantname} --os-target-username={username} --os-target-password="{password}" --os-target-auth-url="{auth_url}" --target_insecure run-action std.noop'.format(tenantname=self.clients.tenant_name, username=self.clients.username, password=self.clients.password, auth_url=self.clients.uri)
        self.mistral_alt_user(cmd=command)