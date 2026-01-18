from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util.semver import SemVer
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class UpgradeAlpha(Upgrade):
    """Upgrade the Kubernetes version of an existing container cluster."""

    @staticmethod
    def Args(parser):
        _Args(parser)
        flags.AddSecurityProfileForUpgradeFlags(parser)

    def ParseUpgradeOptions(self, args):
        ops = ParseUpgradeOptionsBase(args)
        ops.security_profile = args.security_profile
        ops.security_profile_runtime_rules = args.security_profile_runtime_rules
        return ops