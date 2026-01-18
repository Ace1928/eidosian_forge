from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
import atexit
import errno
import os
import sys
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import cli
from googlecloudsdk.command_lib import crash_handling
from googlecloudsdk.command_lib.util.apis import yaml_command_translator
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.credentials import creds_context_managers
from googlecloudsdk.core.credentials import devshell as c_devshell
from googlecloudsdk.core.survey import survey_check
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.util import keyboard_interrupt
from googlecloudsdk.core.util import platforms
import surface
def VersionFunc():
    generated_cli.Execute(['version'])