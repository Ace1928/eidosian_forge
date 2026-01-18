from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from googlecloudsdk.command_lib.code import dataobject
from googlecloudsdk.core import exceptions
def DockerfileAbsPath(self, context):
    return os.path.abspath(os.path.join(context, self.dockerfile))