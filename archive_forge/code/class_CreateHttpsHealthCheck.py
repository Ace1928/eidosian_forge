from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute.https_health_checks import flags
class CreateHttpsHealthCheck(base.CreateCommand):
    """Create a legacy HTTPS health check.

  Though you can use legacy HTTPS health checks in certain Google Cloud Platform
  load balancing configurations and for managed instance group autohealing, you
  should consider a non-legacy HTTPS health check created with `health-checks
  create https` instead.

  For more information about the differences between legacy and non-legacy
  health checks see:
  [](https://cloud.google.com/load-balancing/docs/health-check-concepts#category_and_protocol)

  For information about what type of health check to use for a particular load
  balancer, see:
  [](https://cloud.google.com/load-balancing/docs/health-check-concepts#lb_guide)
  """
    HTTPS_HEALTH_CHECKS_ARG = None

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat(flags.DEFAULT_LIST_FORMAT)
        cls.HTTPS_HEALTH_CHECKS_ARG = flags.HttpsHealthCheckArgument()
        cls.HTTPS_HEALTH_CHECKS_ARG.AddArgument(parser, operation_type='create')
        parser.display_info.AddCacheUpdater(completers.HttpsHealthChecksCompleter)
        parser.add_argument('--host', help='        The value of the host header used in this HTTPS health check request.\n        By default, this is empty and Compute Engine automatically sets\n        the host header in health requests to the same external IP address as\n        the forwarding rule associated with the target pool.\n        ')
        parser.add_argument('--port', type=int, default=443, help='        The TCP port number that this health check monitors. The default value\n        is 443.\n        ')
        parser.add_argument('--request-path', default='/', help="        The request path that this health check monitors. For example,\n        ``/healthcheck''. The default value is ``/''.\n        ")
        parser.add_argument('--check-interval', type=arg_parsers.Duration(), default='5s', help="        How often to perform a health check for an instance. For example,\n        specifying ``10s'' will run the check every 10 seconds. The default\n        value is ``5s''.\n        See $ gcloud topic datetimes for information on duration formats.\n        ")
        parser.add_argument('--timeout', type=arg_parsers.Duration(), default='5s', help="        If Compute Engine doesn't receive an HTTPS 200 response from the\n        instance by the time specified by the value of this flag, the health\n        check request is considered a failure. For example, specifying ``10s''\n        will cause the check to wait for 10 seconds before considering the\n        request a failure. The default value is ``5s''.\n        See $ gcloud topic datetimes for information on duration formats.\n        ")
        parser.add_argument('--unhealthy-threshold', type=int, default=2, help='        The number of consecutive health check failures before a healthy\n        instance is marked as unhealthy. The default is 2.\n        ')
        parser.add_argument('--healthy-threshold', type=int, default=2, help='        The number of consecutive successful health checks before an\n        unhealthy instance is marked as healthy. The default is 2.\n        ')
        parser.add_argument('--description', help='An optional, textual description for the HTTPS health check.')

    def Run(self, args):
        """Issues the request necessary for adding the health check."""
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        health_check_ref = self.HTTPS_HEALTH_CHECKS_ARG.ResolveAsResource(args, holder.resources)
        request = client.messages.ComputeHttpsHealthChecksInsertRequest(httpsHealthCheck=client.messages.HttpsHealthCheck(name=health_check_ref.Name(), host=args.host, port=args.port, description=args.description, requestPath=args.request_path, checkIntervalSec=args.check_interval, timeoutSec=args.timeout, healthyThreshold=args.healthy_threshold, unhealthyThreshold=args.unhealthy_threshold), project=health_check_ref.project)
        return client.MakeRequests([(client.apitools_client.httpsHealthChecks, 'Insert', request)])