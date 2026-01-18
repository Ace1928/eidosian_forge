from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.notebooks import util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def CreateRuntimeDiagnoseRequest(args, messages):
    """"Create and return Diagnose request."""
    runtime = GetRuntimeResource(args).RelativeName()
    diagnostic_config = messages.DiagnosticConfig(gcsBucket=args.gcs_bucket)
    if args.IsSpecified('relative_path'):
        diagnostic_config.relativePath = args.relative_path
    if args.IsSpecified('enable-repair'):
        diagnostic_config.repairFlagEnabled = True
    if args.IsSpecified('enable-packet-capture'):
        diagnostic_config.packetCaptureFlagEnabled = True
    if args.IsSpecified('enable-copy-home-files'):
        diagnostic_config.copyHomeFilesFlagEnabled = True
    timeout_minutes = None
    if args.IsSpecified('timeout_minutes'):
        timeout_minutes = int(args.timeout_minutes)
    return messages.NotebooksProjectsLocationsRuntimesDiagnoseRequest(name=runtime, diagnoseRuntimeRequest=messages.DiagnoseRuntimeRequest(diagnosticConfig=diagnostic_config, timeoutMinutes=timeout_minutes))