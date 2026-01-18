from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import json
import textwrap
from typing import Mapping
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def GetExecutionEnvironment(record):
    return record.annotations.get(k8s_object.EXECUTION_ENVIRONMENT_ANNOTATION, '')