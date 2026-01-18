from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
import six
class CredentialFlags(MutuallyExclusiveGroupDef):

    def AddServiceAccount(self):
        self._AddFlag('--service-account', help='When connecting to Google Cloud Platform services, use a service account key.')

    def AddApplicationDefaultCredential(self):
        self._AddFlag('--application-default-credential', action='store_true', default=False, help='When connecting to Google Cloud Platform services, use the application default credential.')