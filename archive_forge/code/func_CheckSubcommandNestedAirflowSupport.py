from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import random
import re
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def CheckSubcommandNestedAirflowSupport(self, args, airflow_version):
    if args.subcommand_nested and (not image_versions_command_util.IsVersionInRange(airflow_version, '1.10.14', None)):
        raise command_util.Error('Nested subcommands are supported only for Composer environments with Airflow version 1.10.14 or higher.')