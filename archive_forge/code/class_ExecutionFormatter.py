import logging
import os.path
from oslo_serialization import jsonutils
from osc_lib.command import command
from cliff.lister import Lister as cliff_lister
from mistralclient.commands.v2 import base
from mistralclient import utils
class ExecutionFormatter(base.MistralFormatter):
    COLUMNS = [('id', 'ID'), ('workflow_id', 'Workflow ID'), ('workflow_name', 'Workflow name'), ('workflow_namespace', 'Workflow namespace'), ('description', 'Description'), ('task_execution_id', 'Task Execution ID'), ('root_execution_id', 'Root Execution ID'), ('state', 'State'), ('state_info', 'State info'), ('created_at', 'Created at'), ('updated_at', 'Updated at'), ('duration', 'Duration', True)]

    @staticmethod
    def format(wf_ex=None, lister=False):
        if wf_ex:
            state_info = wf_ex.state_info if not lister else base.cut(wf_ex.state_info)
            duration = base.get_duration_str(wf_ex.created_at, wf_ex.updated_at if wf_ex.state in ['ERROR', 'SUCCESS'] else '')
            data = (wf_ex.id, wf_ex.workflow_id, wf_ex.workflow_name, wf_ex.workflow_namespace, wf_ex.description, wf_ex.task_execution_id or '<none>', wf_ex.root_execution_id or '<none>', wf_ex.state, state_info, wf_ex.created_at, wf_ex.updated_at or '<none>', duration)
        else:
            data = (tuple(('' for _ in range(len(ExecutionFormatter.COLUMNS)))),)
        return (ExecutionFormatter.headings(), data)