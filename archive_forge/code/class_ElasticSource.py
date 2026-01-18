from __future__ import (absolute_import, division, print_function)
import getpass
import socket
import time
import uuid
from collections import OrderedDict
from contextlib import closing
from os.path import basename
from ansible.errors import AnsibleError, AnsibleRuntimeError
from ansible.module_utils.six import raise_from
from ansible.plugins.callback import CallbackBase
class ElasticSource(object):

    def __init__(self, display):
        self.ansible_playbook = ''
        self.ansible_version = None
        self.session = str(uuid.uuid4())
        self.host = socket.gethostname()
        try:
            self.ip_address = socket.gethostbyname(socket.gethostname())
        except Exception as e:
            self.ip_address = None
        self.user = getpass.getuser()
        self._display = display

    def start_task(self, tasks_data, hide_task_arguments, play_name, task):
        """ record the start of a task for one or more hosts """
        uuid = task._uuid
        if uuid in tasks_data:
            return
        name = task.get_name().strip()
        path = task.get_path()
        action = task.action
        args = None
        if not task.no_log and (not hide_task_arguments):
            args = ', '.join(('%s=%s' % a for a in task.args.items()))
        tasks_data[uuid] = TaskData(uuid, name, path, play_name, action, args)

    def finish_task(self, tasks_data, status, result):
        """ record the results of a task for a single host """
        task_uuid = result._task._uuid
        if hasattr(result, '_host') and result._host is not None:
            host_uuid = result._host._uuid
            host_name = result._host.name
        else:
            host_uuid = 'include'
            host_name = 'include'
        task = tasks_data[task_uuid]
        if self.ansible_version is None and result._task_fields['args'].get('_ansible_version'):
            self.ansible_version = result._task_fields['args'].get('_ansible_version')
        task.add_host(HostData(host_uuid, host_name, status, result))

    def generate_distributed_traces(self, tasks_data, status, end_time, traceparent, apm_service_name, apm_server_url, apm_verify_server_cert, apm_secret_token, apm_api_key):
        """ generate distributed traces from the collected TaskData and HostData """
        tasks = []
        parent_start_time = None
        for task_uuid, task in tasks_data.items():
            if parent_start_time is None:
                parent_start_time = task.start
            tasks.append(task)
        apm_cli = self.init_apm_client(apm_server_url, apm_service_name, apm_verify_server_cert, apm_secret_token, apm_api_key)
        if apm_cli:
            with closing(apm_cli):
                instrument()
                if traceparent:
                    parent = trace_parent_from_string(traceparent)
                    apm_cli.begin_transaction('Session', trace_parent=parent, start=parent_start_time)
                else:
                    apm_cli.begin_transaction('Session', start=parent_start_time)
                if self.ansible_version is not None:
                    label(ansible_version=self.ansible_version)
                label(ansible_session=self.session, ansible_host_name=self.host, ansible_host_user=self.user)
                if self.ip_address is not None:
                    label(ansible_host_ip=self.ip_address)
                for task_data in tasks:
                    for host_uuid, host_data in task_data.host_data.items():
                        self.create_span_data(apm_cli, task_data, host_data)
                apm_cli.end_transaction(name=__name__, result=status, duration=end_time - parent_start_time)

    def create_span_data(self, apm_cli, task_data, host_data):
        """ create the span with the given TaskData and HostData """
        name = '[%s] %s: %s' % (host_data.name, task_data.play, task_data.name)
        message = 'success'
        status = 'success'
        enriched_error_message = None
        if host_data.status == 'included':
            rc = 0
        else:
            res = host_data.result._result
            rc = res.get('rc', 0)
            if host_data.status == 'failed':
                message = self.get_error_message(res)
                enriched_error_message = self.enrich_error_message(res)
                status = 'failure'
            elif host_data.status == 'skipped':
                if 'skip_reason' in res:
                    message = res['skip_reason']
                else:
                    message = 'skipped'
                status = 'unknown'
        with capture_span(task_data.name, start=task_data.start, span_type='ansible.task.run', duration=host_data.finish - task_data.start, labels={'ansible.task.args': task_data.args, 'ansible.task.message': message, 'ansible.task.module': task_data.action, 'ansible.task.name': name, 'ansible.task.result': rc, 'ansible.task.host.name': host_data.name, 'ansible.task.host.status': host_data.status}) as span:
            span.outcome = status
            if 'failure' in status:
                exception = AnsibleRuntimeError(message='{0}: {1} failed with error message {2}'.format(task_data.action, name, enriched_error_message))
                apm_cli.capture_exception(exc_info=(type(exception), exception, exception.__traceback__), handled=True)

    def init_apm_client(self, apm_server_url, apm_service_name, apm_verify_server_cert, apm_secret_token, apm_api_key):
        if apm_server_url:
            return Client(service_name=apm_service_name, server_url=apm_server_url, verify_server_cert=False, secret_token=apm_secret_token, api_key=apm_api_key, use_elastic_traceparent_header=True, debug=True)

    @staticmethod
    def get_error_message(result):
        if result.get('exception') is not None:
            return ElasticSource._last_line(result['exception'])
        return result.get('msg', 'failed')

    @staticmethod
    def _last_line(text):
        lines = text.strip().split('\n')
        return lines[-1]

    @staticmethod
    def enrich_error_message(result):
        message = result.get('msg', 'failed')
        exception = result.get('exception')
        stderr = result.get('stderr')
        return 'message: "{0}"\nexception: "{1}"\nstderr: "{2}"'.format(message, exception, stderr)