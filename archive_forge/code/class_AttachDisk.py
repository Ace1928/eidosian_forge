from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import resource_args
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AttachDisk(base.Command):
    """Attach disk to TPU VM."""

    @staticmethod
    def Args(parser):
        """Set up arguments for this command.

    Args:
      parser: An argparse.ArgumentParser.
    """
        parser.add_argument('--disk', help='The name of the disk to attach to the TPU VM.', required=True)
        parser.add_argument('--mode', choices=MODE_OPTIONS, default='read-write', help='Specifies the mode of the disk.')
        AddTPUResourceArg(parser, 'to attach disk')

    def Run(self, args):
        if args.zone is None:
            args.zone = properties.VALUES.compute.zone.Get(required=True)
        tpu_name_ref = args.CONCEPTS.tpu.Parse()
        tpu = tpu_utils.TPUNode(self.ReleaseTrack())
        node = tpu.Get(tpu_name_ref.Name(), args.zone)
        if not tpu_utils.IsTPUVMNode(node):
            raise exceptions.BadArgumentException('TPU', 'this command is only available for Cloud TPU VM nodes. To access this node, please see https://cloud.google.com/tpu/docs/creating-deleting-tpus.')
        if args.mode == 'read-write':
            args.mode = tpu.messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE
        elif args.mode == 'read-only':
            args.mode = tpu.messages.AttachedDisk.ModeValueValuesEnum.READ_ONLY
        else:
            raise exceptions.BadArgumentException('mode', 'can only attach disks in read-write or read-only mode.')
        source_path_match = re.match('projects/.+/(zones|regions)/.+/disks/.+', args.disk)
        if source_path_match:
            source_path = args.disk
        else:
            project = properties.VALUES.core.project.Get(required=True)
            source_path = 'projects/' + project + '/zones/' + args.zone + '/disks/' + args.disk
        if not node.dataDisks:
            disk_to_attach = tpu.messages.AttachedDisk(mode=args.mode, sourceDisk=source_path)
            node.dataDisks.append(disk_to_attach)
        else:
            source_disk_list = []
            for disk in node.dataDisks:
                source_disk_list.append(disk.sourceDisk)
            if source_path not in source_disk_list:
                disk_to_attach = tpu.messages.AttachedDisk(mode=args.mode, sourceDisk=source_path)
                node.dataDisks.append(disk_to_attach)
            else:
                raise exceptions.BadArgumentException('TPU', 'disk is already attached to the TPU VM.')
        return tpu.UpdateNode(name=tpu_name_ref.Name(), zone=args.zone, node=node, update_mask='data_disks', poller_message='Attaching disk to TPU VM')