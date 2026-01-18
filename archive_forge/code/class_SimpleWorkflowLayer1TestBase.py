import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
class SimpleWorkflowLayer1TestBase(unittest.TestCase):
    """
    There are at least two test cases which share this setUp/tearDown
    and the class-based parameter definitions:
        * SimpleWorkflowLayer1Test
        * tests.swf.test_layer1_workflow_execution.SwfL1WorkflowExecutionTest
    """
    swf = True
    _domain = BOTO_SWF_UNITTEST_DOMAIN
    _workflow_execution_retention_period_in_days = 'NONE'
    _domain_description = 'test workflow domain'
    _task_list = 'tasklist1'
    _workflow_type_name = 'wft1'
    _workflow_type_version = '1'
    _workflow_type_description = 'wft1 description'
    _default_child_policy = 'REQUEST_CANCEL'
    _default_execution_start_to_close_timeout = '600'
    _default_task_start_to_close_timeout = '60'
    _activity_type_name = 'at1'
    _activity_type_version = '1'
    _activity_type_description = 'at1 description'
    _default_task_heartbeat_timeout = '30'
    _default_task_schedule_to_close_timeout = '90'
    _default_task_schedule_to_start_timeout = '10'
    _default_task_start_to_close_timeout = '30'

    def setUp(self):
        self.conn = Layer1()
        try:
            r = self.conn.register_domain(self._domain, self._workflow_execution_retention_period_in_days, description=self._domain_description)
            assert r is None
            time.sleep(PAUSE_SECONDS)
        except swf_exceptions.SWFDomainAlreadyExistsError:
            pass
        try:
            r = self.conn.register_workflow_type(self._domain, self._workflow_type_name, self._workflow_type_version, task_list=self._task_list, default_child_policy=self._default_child_policy, default_execution_start_to_close_timeout=self._default_execution_start_to_close_timeout, default_task_start_to_close_timeout=self._default_task_start_to_close_timeout, description=self._workflow_type_description)
            assert r is None
            time.sleep(PAUSE_SECONDS)
        except swf_exceptions.SWFTypeAlreadyExistsError:
            pass
        try:
            r = self.conn.register_activity_type(self._domain, self._activity_type_name, self._activity_type_version, task_list=self._task_list, default_task_heartbeat_timeout=self._default_task_heartbeat_timeout, default_task_schedule_to_close_timeout=self._default_task_schedule_to_close_timeout, default_task_schedule_to_start_timeout=self._default_task_schedule_to_start_timeout, default_task_start_to_close_timeout=self._default_task_start_to_close_timeout, description=self._activity_type_description)
            assert r is None
            time.sleep(PAUSE_SECONDS)
        except swf_exceptions.SWFTypeAlreadyExistsError:
            pass

    def tearDown(self):
        pass