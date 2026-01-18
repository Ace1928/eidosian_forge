from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.compute.networks import flags as compute_network_flags
from googlecloudsdk.command_lib.compute.networks.subnets import flags as compute_subnet_flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.workbench import completers
from googlecloudsdk.core import properties
def AddDiagnosticConfigFlags(parser, vm_type):
    """Adds Diagnostic config flags to parser."""
    parser.add_argument('--gcs-bucket', dest='gcs_bucket', help="The Cloud Storage bucket where the log files generated from the diagnose command will be stored. storage.buckets.writer permissions must be given to project's service account or user credential. Format: gs://`{gcs_bucket}` ", required=True)
    parser.add_argument('--relative-path', dest='relative_path', help='Defines the relative storage path in the Cloud Storage bucket where the diagnostic logs will be written. Default path will be the root directory of the Cloud Storage bucketFormat of full path: gs://`{gcs_bucket}`/`{relative_path}`/ ', required=False)
    parser.add_argument('--enable-repair', action='store_true', dest='enable_repair', default=False, help='Enables flag to repair service for {}'.format(vm_type), required=False)
    parser.add_argument('--enable-packet-capture', action='store_true', dest='enable_packet_capture', default=False, help='Enables flag to capture packets from the {} for 30 seconds'.format(vm_type), required=False)
    parser.add_argument('--enable-copy-home-files', action='store_true', dest='enable_copy_home_files', default=False, help='Enables flag to copy all `/home/jupyter` folder contents', required=False)