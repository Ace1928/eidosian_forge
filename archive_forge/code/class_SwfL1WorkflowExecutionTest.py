import time
import uuid
import json
import traceback
from boto.swf.layer1_decisions import Layer1Decisions
from tests.integration.swf.test_layer1 import SimpleWorkflowLayer1TestBase
class SwfL1WorkflowExecutionTest(SimpleWorkflowLayer1TestBase):
    """
    test a simple workflow execution
    """
    swf = True

    def run_decider(self):
        """
        run one iteration of a simple decision engine
        """
        tries = 0
        while True:
            dtask = self.conn.poll_for_decision_task(self._domain, self._task_list, reverse_order=True)
            if dtask.get('taskToken') is not None:
                break
            time.sleep(2)
            tries += 1
            if tries > 10:
                assert False, 'no decision task occurred'
        ignorable = ('DecisionTaskScheduled', 'DecisionTaskStarted', 'DecisionTaskTimedOut')
        event = None
        for tevent in dtask['events']:
            if tevent['eventType'] not in ignorable:
                event = tevent
                break
        decisions = Layer1Decisions()
        if event['eventType'] == 'WorkflowExecutionStarted':
            activity_id = str(uuid.uuid1())
            decisions.schedule_activity_task(activity_id, self._activity_type_name, self._activity_type_version, task_list=self._task_list, input=event['workflowExecutionStartedEventAttributes']['input'])
        elif event['eventType'] == 'ActivityTaskCompleted':
            decisions.complete_workflow_execution(result=event['activityTaskCompletedEventAttributes']['result'])
        elif event['eventType'] == 'ActivityTaskFailed':
            decisions.fail_workflow_execution(reason=event['activityTaskFailedEventAttributes']['reason'], details=event['activityTaskFailedEventAttributes']['details'])
        else:
            decisions.fail_workflow_execution(reason='unhandled decision task type; %r' % (event['eventType'],))
        r = self.conn.respond_decision_task_completed(dtask['taskToken'], decisions=decisions._data, execution_context=None)
        assert r is None

    def run_worker(self):
        """
        run one iteration of a simple worker engine
        """
        tries = 0
        while True:
            atask = self.conn.poll_for_activity_task(self._domain, self._task_list, identity='test worker')
            if atask.get('activityId') is not None:
                break
            time.sleep(2)
            tries += 1
            if tries > 10:
                assert False, 'no activity task occurred'
        reason = None
        try:
            result = json.dumps(sum(json.loads(atask['input'])))
        except:
            reason = 'an exception was raised'
            details = traceback.format_exc()
        if reason is None:
            r = self.conn.respond_activity_task_completed(atask['taskToken'], result)
        else:
            r = self.conn.respond_activity_task_failed(atask['taskToken'], reason=reason, details=details)
        assert r is None

    def test_workflow_execution(self):
        workflow_id = 'wfid-%.2f' % (time.time(),)
        r = self.conn.start_workflow_execution(self._domain, workflow_id, self._workflow_type_name, self._workflow_type_version, execution_start_to_close_timeout='20', input='[600, 15]')
        run_id = r['runId']
        self.run_decider()
        self.run_worker()
        self.run_decider()
        r = self.conn.get_workflow_execution_history(self._domain, run_id, workflow_id, reverse_order=True)['events'][0]
        result = r['workflowExecutionCompletedEventAttributes']['result']
        assert json.loads(result) == 615

    def test_failed_workflow_execution(self):
        workflow_id = 'wfid-%.2f' % (time.time(),)
        r = self.conn.start_workflow_execution(self._domain, workflow_id, self._workflow_type_name, self._workflow_type_version, execution_start_to_close_timeout='20', input='[600, "s"]')
        run_id = r['runId']
        self.run_decider()
        self.run_worker()
        self.run_decider()
        r = self.conn.get_workflow_execution_history(self._domain, run_id, workflow_id, reverse_order=True)['events'][0]
        reason = r['workflowExecutionFailedEventAttributes']['reason']
        assert reason == 'an exception was raised'