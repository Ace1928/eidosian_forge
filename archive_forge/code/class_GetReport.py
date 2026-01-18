import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
class GetReport(command.Command):
    """Print execution report."""

    def get_parser(self, prog_name):
        parser = super(GetReport, self).get_parser(prog_name)
        parser.add_argument('id', help='Execution ID')
        parser.add_argument('--errors-only', dest='errors_only', action='store_true', help='Only error paths will be included.')
        parser.add_argument('--statistics-only', dest='statistics_only', action='store_true', help='Only the statistics will be included.')
        parser.add_argument('--no-errors-only', dest='errors_only', action='store_false', help='Not only error paths will be included.')
        parser.set_defaults(errors_only=True)
        parser.add_argument('--max-depth', dest='max_depth', nargs='?', type=int, default=-1, help='Maximum depth of the workflow execution tree. If 0, only the root workflow execution and its tasks will be included')
        return parser

    def print_line(self, line, level=0):
        self.app.stdout.write('%s%s\n' % (' ' * (level * REPORT_ENTRY_INDENT), line))

    def print_workflow_execution_entry(self, wf_ex, level):
        self.print_line("workflow '%s' [%s] %s" % (wf_ex['name'], wf_ex['state'], wf_ex['id']), level)
        if 'task_executions' in wf_ex:
            for t_ex in wf_ex['task_executions']:
                self.print_task_execution_entry(t_ex, level + 1)

    def print_task_execution_entry(self, t_ex, level):
        self.print_line("task '%s' [%s] %s" % (t_ex['name'], t_ex['state'], t_ex['id']), level)
        if 'retry_count' in t_ex:
            self.print_line('(retry count: %s)' % t_ex['retry_count'], level)
        if t_ex['state'] == 'ERROR':
            state_info = t_ex['state_info']
            if state_info:
                state_info = state_info[0:100].replace('\n', ' ') + '...'
                self.print_line('(error info: %s)' % state_info, level)
        if 'action_executions' in t_ex:
            for a_ex in t_ex['action_executions']:
                self.print_action_execution_entry(a_ex, level + 1)
        if 'workflow_executions' in t_ex:
            for wf_ex in t_ex['workflow_executions']:
                self.print_workflow_execution_entry(wf_ex, level + 1)

    def print_action_execution_entry(self, a_ex, level):
        self.print_line("action '%s' [%s] %s" % (a_ex['name'], a_ex['state'], a_ex['id']), level)
        if a_ex['state'] == 'ERROR':
            state_info = a_ex['state_info']
            if state_info:
                state_info = state_info[0:100] + '...'
                self.print_line('(error info: %s)' % state_info, level)

    def print_statistics(self, stat):
        self.print_line('Number of tasks in SUCCESS state: %s' % stat['success_tasks_count'])
        self.print_line('Number of tasks in ERROR state: %s' % stat['error_tasks_count'])
        self.print_line('Number of tasks in RUNNING state: %s' % stat['running_tasks_count'])
        self.print_line('Number of tasks in IDLE state: %s' % stat['idle_tasks_count'])
        self.print_line('Number of tasks in PAUSED state: %s\n' % stat['paused_tasks_count'])
        if 'estimated_time' in stat:
            self.print_line('Estimated time (seconds) for the execution to finish: %s\n' % stat['estimated_time'])

    def print_report(self, report_json):
        self.print_line("\nTo get more details on a task failure run: mistral task-get <id> -c 'State info'\n")
        frame_line = '=' * 30
        self.print_line('%s General Statistics %s\n' % (frame_line, frame_line))
        self.print_statistics(report_json['statistics'])
        if 'root_workflow_execution' in report_json:
            self.print_line('%s Workflow Execution Tree %s\n' % (frame_line, frame_line))
            self.print_workflow_execution_entry(report_json['root_workflow_execution'], 0)

    def take_action(self, parsed_args):
        mistral_client = self.app.client_manager.workflow_engine
        report_json = mistral_client.executions.get_report(parsed_args.id, errors_only=parsed_args.errors_only, max_depth=parsed_args.max_depth, statistics_only=parsed_args.statistics_only)
        self.print_report(report_json)