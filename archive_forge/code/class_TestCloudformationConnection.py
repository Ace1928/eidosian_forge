import time
import json
from tests.unit import  unittest
from boto.cloudformation.connection import CloudFormationConnection
class TestCloudformationConnection(unittest.TestCase):

    def setUp(self):
        self.connection = CloudFormationConnection()
        self.stack_name = 'testcfnstack' + str(int(time.time()))

    def test_large_template_stack_size(self):
        body = self.connection.create_stack(self.stack_name, template_body=json.dumps(BASIC_EC2_TEMPLATE), parameters=[('Parameter1', 'initial_value'), ('Parameter2', 'initial_value')])
        self.addCleanup(self.connection.delete_stack, self.stack_name)
        events = self.connection.describe_stack_events(self.stack_name)
        self.assertTrue(events)
        policy = self.connection.get_stack_policy(self.stack_name)
        self.assertEqual(None, policy)
        stacks = self.connection.describe_stacks(self.stack_name)
        stack = stacks[0]
        self.assertEqual(self.stack_name, stack.stack_name)
        params = [(p.key, p.value) for p in stack.parameters]
        self.assertEquals([('Parameter1', 'initial_value'), ('Parameter2', 'initial_value')], params)
        for _ in range(30):
            stack.update()
            if stack.stack_status.find('PROGRESS') == -1:
                break
            time.sleep(5)
        body = self.connection.update_stack(self.stack_name, template_body=json.dumps(BASIC_EC2_TEMPLATE), parameters=[('Parameter1', '', True), ('Parameter2', 'updated_value')])
        stacks = self.connection.describe_stacks(self.stack_name)
        stack = stacks[0]
        params = [(p.key, p.value) for p in stacks[0].parameters]
        self.assertEquals([('Parameter1', 'initial_value'), ('Parameter2', 'updated_value')], params)
        for _ in range(30):
            stack.update()
            if stack.stack_status.find('PROGRESS') == -1:
                break
            time.sleep(5)