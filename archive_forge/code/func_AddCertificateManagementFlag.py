from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def AddCertificateManagementFlag(parser):
    """Adds common flags to a domain-mappings command."""
    certificate_argument = base.ChoiceArgument('--certificate-management', choices=['automatic', 'manual'], help_str="Type of certificate management. 'automatic' will provision an SSL certificate automatically while 'manual' requires the user to provide a certificate id to provision.")
    certificate_argument.AddToParser(parser)