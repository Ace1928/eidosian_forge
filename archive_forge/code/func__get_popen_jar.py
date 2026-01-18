from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import execution_utils
def _get_popen_jar(appname):
    if appname not in samples.APPS:
        raise ValueError("Unknown sample app '{}'".format(appname))
    return os.path.join(samples.get_local_bin_path(appname), samples.APPS[appname].workload_bin)