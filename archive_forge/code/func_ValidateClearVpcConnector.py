from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.core.console import console_io
def ValidateClearVpcConnector(service, args):
    """Validates that the VPC connector can be safely removed.

  Does nothing if 'clear_vpc_connector' is not present in args with value True.

  Args:
    service: A Cloud Run service object.
    args: Namespace object containing the specified command line arguments.

  Raises:
    exceptions.ConfigurationError: If the command cannot prompt and
      VPC egress is set to 'all' or 'all-traffic'.
    console_io.OperationCancelledError: If the user answers no to the
      confirmation prompt.
  """
    if service is None or not flags.FlagIsExplicitlySet(args, 'clear_vpc_connector') or (not args.clear_vpc_connector):
        return
    if flags.FlagIsExplicitlySet(args, 'vpc_egress'):
        egress = args.vpc_egress
    elif container_resource.EGRESS_SETTINGS_ANNOTATION in service.template_annotations:
        egress = service.template_annotations[container_resource.EGRESS_SETTINGS_ANNOTATION]
    else:
        return
    if egress != container_resource.EGRESS_SETTINGS_ALL and egress != container_resource.EGRESS_SETTINGS_ALL_TRAFFIC:
        return
    if console_io.CanPrompt():
        console_io.PromptContinue(message='Removing the VPC connector from this service will clear the VPC egress setting and route outbound traffic to the public internet.', default=False, cancel_on_no=True)
    else:
        raise exceptions.ConfigurationError('Cannot remove VPC connector with VPC egress set to "{}". Set `--vpc-egress=private-ranges-only` or run this command interactively and provide confirmation to continue.'.format(egress))