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
def ServiceAccountFlag():
    return StringFlag('--service-account', help='Service account associated with the revision of the service. The service account represents the identity of the running revision, and determines what permissions the revision has. This is the name of a Kubernetes service account in the same namespace as the service. If not provided, the revision will use the default Kubernetes namespace service account. To reset this field to its default, pass an empty string.')