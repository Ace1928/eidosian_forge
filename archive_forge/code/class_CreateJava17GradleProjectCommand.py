from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import os
import re
import shutil
import tempfile
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.command_lib.app import jarfile
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class CreateJava17GradleProjectCommand(CreateJava17ProjectCommand):
    """A command that creates a java17 runtime app.yaml from a build.gradle file."""

    def __init__(self):
        self.error = GradleBuildNotSupported
        self.ignore = 'build'
        super(CreateJava17GradleProjectCommand, self).__init__()

    def __eq__(self, other):
        return isinstance(other, CreateJava17GradleProjectCommand)