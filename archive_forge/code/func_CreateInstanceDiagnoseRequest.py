from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.workbench import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateInstanceDiagnoseRequest(args, messages):
    """"Create and return Diagnose request."""
    instance = GetInstanceResource(args).RelativeName()
    diagnostic_config = messages.DiagnosticConfig(gcsBucket=args.gcs_bucket)
    if args.IsSpecified('relative_path'):
        diagnostic_config.relativePath = args.relative_path
    if args.IsSpecified('enable_repair'):
        diagnostic_config.enableRepairFlag = True
    if args.IsSpecified('enable_packet_capture'):
        diagnostic_config.enablePacketCaptureFlag = True
    if args.IsSpecified('enable_copy_home_files'):
        diagnostic_config.enableCopyHomeFilesFlag = True
    return messages.NotebooksProjectsLocationsInstancesDiagnoseRequest(name=instance, diagnoseInstanceRequest=messages.DiagnoseInstanceRequest(diagnosticConfig=diagnostic_config))