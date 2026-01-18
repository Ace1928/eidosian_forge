import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
class SimpleMistralCLITests(base.MistralCLIAuth):
    """Basic tests, check '-list', '-help' commands."""
    _mistral_url = MISTRAL_URL

    @classmethod
    def setUpClass(cls):
        super(SimpleMistralCLITests, cls).setUpClass()

    def test_workbooks_list(self):
        workbooks = self.parser.listing(self.mistral('workbook-list'))
        self.assertTableStruct(workbooks, ['Name', 'Tags', 'Created at', 'Updated at'])

    def test_workflow_list(self):
        workflows = self.parser.listing(self.mistral('workflow-list'))
        self.assertTableStruct(workflows, ['ID', 'Name', 'Tags', 'Input', 'Scope', 'Created at', 'Updated at'])

    def test_executions_list(self):
        executions = self.parser.listing(self.mistral('execution-list'))
        self.assertTableStruct(executions, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])

    def test_tasks_list(self):
        tasks = self.parser.listing(self.mistral('task-list'))
        self.assertTableStruct(tasks, ['ID', 'Name', 'Workflow name', 'Workflow Execution ID', 'State'])

    def test_cron_trigger_list(self):
        triggers = self.parser.listing(self.mistral('cron-trigger-list'))
        self.assertTableStruct(triggers, ['Name', 'Workflow', 'Pattern', 'Next execution time', 'Remaining executions', 'Created at', 'Updated at'])

    def test_event_trigger_list(self):
        triggers = self.parser.listing(self.mistral('event-trigger-list'))
        self.assertTableStruct(triggers, ['ID', 'Name', 'Workflow ID', 'Exchange', 'Topic', 'Event', 'Created at', 'Updated at'])

    def test_actions_list(self):
        actions = self.parser.listing(self.mistral('action-list'))
        self.assertTableStruct(actions, ['Name', 'Is system', 'Input', 'Description', 'Tags', 'Created at', 'Updated at'])

    def test_environments_list(self):
        envs = self.parser.listing(self.mistral('environment-list'))
        self.assertTableStruct(envs, ['Name', 'Description', 'Scope', 'Created at', 'Updated at'])

    def test_action_execution_list(self):
        act_execs = self.parser.listing(self.mistral('action-execution-list'))
        self.assertTableStruct(act_execs, ['ID', 'Name', 'Workflow name', 'State', 'Accepted'])

    def test_action_execution_list_with_limit(self):
        act_execs = self.parser.listing(self.mistral('action-execution-list', params='--limit 1'))
        self.assertEqual(1, len(act_execs))