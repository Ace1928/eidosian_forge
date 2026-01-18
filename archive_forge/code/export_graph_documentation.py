from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run.integrations import graph
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
from googlecloudsdk.core import log
This method is called to print the result of the Run() method.

    Args:
      args: all the arguments that were provided to this command invocation.
      bindings: The binding data returned from Run().
    