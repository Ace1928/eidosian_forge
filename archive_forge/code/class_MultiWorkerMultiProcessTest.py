import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import unittest
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training import server_lib
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class MultiWorkerMultiProcessTest(test.TestCase):
    """Testing infra for independent workers using multiple processes."""

    def _run_task_in_process(self, cmd_args, cluster_spec, task_type, task_id):
        env = os.environ.copy()
        env['TF_CONFIG'] = json.dumps({'cluster': cluster_spec, 'task': {'type': task_type, 'index': task_id}})
        return subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    @deprecation.deprecated(None, '`run_multiple_tasks_in_processes` is deprecated; any new test requiring multiple processes should use `multi_process_runner` for better support of log printing, streaming, and more functionality.')
    def run_multiple_tasks_in_processes(self, cmd_args, cluster_spec):
        """Run `cmd_args` in a process for each task in `cluster_spec`."""
        processes = {}
        for task_type in cluster_spec.keys():
            processes[task_type] = []
            for task_id in range(len(cluster_spec[task_type])):
                p = self._run_task_in_process(cmd_args, cluster_spec, task_type, task_id)
                processes[task_type].append(p)
        return processes

    @deprecation.deprecated(None, '`join_independent_workers` is deprecated; any new test requiring multiple processes should use `multi_process_runner` for better support of log printing, streaming, and more functionality.')
    def join_independent_workers(self, worker_processes):
        return_codes = []
        for p in nest.flatten(worker_processes):
            try:
                p.communicate()
            except ValueError:
                pass
            finally:
                return_codes.append(p.returncode)
        for return_code in return_codes:
            self.assertEqual(return_code, 0)

    @deprecation.deprecated(None, '`stream_stderr` is deprecated; any new test requiring multiple processes should use `multi_process_runner` for better support of log printing, streaming, and more functionality.')
    def stream_stderr(self, processes, print_only_first=False):
        """Consume stderr of all processes and print to stdout.

    To reduce the amount of logging, caller can set print_only_first to True.
    In that case, this function only prints stderr from the first process of
    each type.

    Args:
      processes: A dictionary from process type string -> list of processes.
      print_only_first: If true, only print output from first process of each
        type.
    """

        def _stream_stderr_single_process(process, type_string, index, print_to_stdout):
            """Consume a single process's stderr and optionally print to stdout."""
            while True:
                output = process.stderr.readline()
                if not output and process.poll() is not None:
                    break
                if output and print_to_stdout:
                    print('{}{} {}'.format(type_string, index, output.strip()))
                    sys.stdout.flush()
        stream_threads = []
        for process_type, process_list in six.iteritems(processes):
            for i in range(len(process_list)):
                print_to_stdout = not print_only_first or i == 0
                thread = threading.Thread(target=_stream_stderr_single_process, args=(process_list[i], process_type, i, print_to_stdout))
                thread.start()
                stream_threads.append(thread)
        for thread in stream_threads:
            thread.join()