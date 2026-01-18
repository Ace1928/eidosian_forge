from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def NoTrafficFlag():
    return BasicFlag('--no-traffic', help='If set, any traffic assigned to the LATEST revision will be assigned to the specific revision bound to LATEST before the deployment. This means the revision being deployed will not receive traffic. After a deployment with this flag, the LATEST revision will not receive traffic on future deployments.')