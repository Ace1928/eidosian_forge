from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instances import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
class TailSerialPortOutput(base.Command):
    """Periodically fetch new output from a virtual machine instance's serial port and display it as it becomes available.

  {command} is used to tail the output from a Compute
  Engine virtual machine instance's serial port. The serial port output
  from the instance will be printed to standard output. This
  information can be useful for diagnostic purposes.
  """
    detailed_help = {'EXAMPLES': "\n  To fetch new output from instance's serial port and display it, run:\n\n    $ {command} example-instance --zone=us-central1-b\n  "}
    POLL_SLEEP_SECS = 10

    @staticmethod
    def Args(parser):
        flags.INSTANCE_ARG.AddArgument(parser)
        parser.add_argument('--port', type=arg_parsers.BoundedInt(1, 4), help='        Instances can support up to four serial port outputs, numbered 1 through\n        4. By default, this command will return the output of the first serial\n        port. Setting this flag will return the output of the requested serial\n        port.\n        ')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        instance_ref = flags.INSTANCE_ARG.ResolveAsResource(args, holder.resources, scope_lister=flags.GetInstanceZoneScopeLister(client))
        start = None
        while True:
            request = (client.apitools_client.instances, 'GetSerialPortOutput', client.messages.ComputeInstancesGetSerialPortOutputRequest(instance=instance_ref.Name(), project=instance_ref.project, port=args.port, start=start, zone=instance_ref.zone))
            errors = []
            objects = client.MakeRequests(requests=[request], errors_to_collect=errors)
            if errors:
                raise TailSerialPortOutputException('Could not fetch serial port output: ' + ','.join([error[1] for error in errors]))
            result = objects[0]
            log.out.write(result.contents)
            start = result.next
            if not result.contents:
                time.sleep(self.POLL_SLEEP_SECS)