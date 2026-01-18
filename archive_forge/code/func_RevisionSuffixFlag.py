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
def RevisionSuffixFlag():
    return StringFlag('--revision-suffix', help="Suffix of the revision name. Revision names always start with the service name automatically. For example, specifying `--revision-suffix=v1` for a service named 'helloworld', would lead to a revision named 'helloworld-v1'.")