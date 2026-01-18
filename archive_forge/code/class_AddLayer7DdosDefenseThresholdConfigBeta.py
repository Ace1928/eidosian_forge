from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.security_policies import flags
from googlecloudsdk.command_lib.compute.security_policies import security_policies_utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class AddLayer7DdosDefenseThresholdConfigBeta(base.UpdateCommand):
    """Add a layer7 ddos defense threshold config to a Compute Engine security policy.

  *{command}* is used to add layer7 ddos defense threshold configs to security policies.

  ## EXAMPLES

  To add a layer7 ddos defense threshold config run the following command:

    $ {command} NAME \\
       --threshold-config-name=my-threshold-config-name \\
       --auto-deploy-load-threshold=0.7 \\
       --auto-deploy-confidence-threshold=0.8 \\
       --auto-deploy-impacted-baseline-threshold=0.1 \\
       --auto-deploy-expiration-sec=4800
       --detection-load-threshold=0.4
       --detection-absolute-qps=1000
       --detection-relative-to-baseline-qps=2.0
       --traffic-granularity-configs=type=HTTP_HEADER_HOST,value=www.my-test-host.com;type=HTTP_PATH,enableEachUniqueValue=true
  """
    _support_granularity_config = True

    @classmethod
    def Args(cls, parser):
        AddLayer7DdosDefenseThresholdConfigHelper.Args(parser, support_granularity_config=cls._support_granularity_config)

    def Run(self, args):
        return AddLayer7DdosDefenseThresholdConfigHelper.Run(self.ReleaseTrack(), args, support_granularity_config=self._support_granularity_config)