from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def RunKubectlCommand(args, out_func=None, err_func=None, namespace=None):
    """Shells out a command to kubectl.

  This command should be called within the context of a TemporaryKubeconfig
  context manager in order for kubectl to be configured to access the correct
  cluster.

  Args:
    args: list of strings, command line arguments to pass to the kubectl
      command. Should omit the kubectl command itself. For example, to execute
      'kubectl get pods', provide ['get', 'pods'].
    out_func: str->None, a function to call with the stdout of the kubectl
      command
    err_func: str->None, a function to call with the stderr of the kubectl
      command
    namespace: str or None, the kubectl namespace to apply to the command

  Raises:
    Error: if kubectl could not be called
    KubectlError: if the invocation of kubectl was unsuccessful
  """
    kubectl_path = files.FindExecutableOnPath(_KUBECTL_COMPONENT_NAME, config.Paths().sdk_bin_path)
    if kubectl_path is None:
        kubectl_path = files.FindExecutableOnPath(_KUBECTL_COMPONENT_NAME)
    if kubectl_path is None:
        raise Error(MISSING_KUBECTL_MSG)
    exec_args = AddKubectlNamespace(namespace, execution_utils.ArgsForExecutableTool(kubectl_path, *args))
    try:
        retval = execution_utils.Exec(exec_args, no_exit=True, out_func=out_func, err_func=lambda err: HandleKubectlErrorStream(err_func, err), universal_newlines=True)
    except (execution_utils.PermissionError, execution_utils.InvalidCommandError) as e:
        raise KubectlError(six.text_type(e))
    if retval:
        raise KubectlError('kubectl returned non-zero status code.')