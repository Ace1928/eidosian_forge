from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import display
from googlecloudsdk.command_lib.run import config_changes
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.command_lib.run import resource_args
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run import stages
from googlecloudsdk.command_lib.run.printers import traffic_printer
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
@base.ReleaseTracks(base.ReleaseTrack.GA)
class AdjustTraffic(base.Command):
    """Adjust the traffic assignments for a Cloud Run service."""
    detailed_help = {'DESCRIPTION': '          {description}\n          ', 'EXAMPLES': '          To assign 10% of traffic to revision myservice-s5sxn and\n          90% of traffic to revision myservice-cp9kw run:\n\n              $ {command} myservice --to-revisions=myservice-s5sxn=10,myservice-cp9kw=90\n\n          To increase the traffic to revision myservice-s5sxn to 20% and\n          by reducing the traffic to revision myservice-cp9kw to 80% run:\n\n              $ {command} myservice --to-revisions=myservice-s5sxn=20\n\n          To rollback to revision myservice-cp9kw run:\n\n              $ {command} myservice --to-revisions=myservice-cp9kw=100\n\n          To assign 100% of traffic to the current or future LATEST revision\n          run:\n\n              $ {command} myservice --to-latest\n\n          You can also refer to the current or future LATEST revision in\n          --to-revisions by the string "LATEST". For example, to set 10% of\n          traffic to always float to the latest revision:\n\n              $ {command} myservice --to-revisions=LATEST=10\n\n         '}

    @classmethod
    def CommonArgs(cls, parser):
        service_presentation = presentation_specs.ResourcePresentationSpec('SERVICE', resource_args.GetServiceResourceSpec(prompt=True), 'Service to update the configuration of.', required=True, prefixes=False)
        flags.AddAsyncFlag(parser)
        flags.AddUpdateTrafficFlags(parser)
        flags.AddTrafficTagsFlags(parser)
        concept_parsers.ConceptParser([service_presentation]).AddToParser(parser)
        resource_printer.RegisterFormatter(traffic_printer.TRAFFIC_PRINTER_FORMAT, traffic_printer.TrafficPrinter, hidden=True)
        parser.display_info.AddFormat(traffic_printer.TRAFFIC_PRINTER_FORMAT)

    @classmethod
    def Args(cls, parser):
        cls.CommonArgs(parser)

    def Run(self, args):
        """Update the traffic split for the service.

    Args:
      args: Args!

    Returns:
      List of traffic.TrafficTargetStatus instances reflecting the change.
    """
        conn_context = connection_context.GetConnectionContext(args, flags.Product.RUN, self.ReleaseTrack())
        service_ref = args.CONCEPTS.service.Parse()
        flags.ValidateResource(service_ref)
        changes = flags.GetServiceConfigurationChanges(args)
        if not changes:
            raise exceptions.NoConfigurationChangeError('No traffic configuration change requested.')
        changes.insert(0, config_changes.DeleteAnnotationChange(k8s_object.BINAUTHZ_BREAKGLASS_ANNOTATION))
        changes.append(config_changes.SetLaunchStageAnnotationChange(self.ReleaseTrack()))
        is_managed = platforms.GetPlatform() == platforms.PLATFORM_MANAGED
        with serverless_operations.Connect(conn_context) as client:
            deployment_stages = stages.UpdateTrafficStages()
            try:
                with progress_tracker.StagedProgressTracker('Updating traffic...', deployment_stages, failure_message='Updating traffic failed', suppress_output=args.async_) as tracker:
                    serv = client.UpdateTraffic(service_ref, changes, tracker, args.async_)
            except:
                serv = client.GetService(service_ref)
                if serv:
                    resources = traffic_pair.GetTrafficTargetPairs(serv.spec_traffic, serv.status_traffic, is_managed, serv.status.latestReadyRevisionName, serv.status.url)
                    display.Displayer(self, args, resources, display_info=args.GetDisplayInfo()).Display()
                raise
            if args.async_:
                pretty_print.Success('Updating traffic asynchronously.')
            else:
                resources = traffic_pair.GetTrafficTargetPairs(serv.spec_traffic, serv.status_traffic, is_managed, serv.status.latestReadyRevisionName, serv.status.url)
                return resources