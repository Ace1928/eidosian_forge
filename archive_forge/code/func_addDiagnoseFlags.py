from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import retry
@staticmethod
def addDiagnoseFlags(parser, dataproc):
    parser.add_argument('--tarball-access', type=arg_utils.ChoiceToEnumName, choices=Diagnose._GetValidTarballAccessChoices(dataproc), help='Target access privileges for diagnose tarball.')
    parser.add_argument('--start-time', help='Time instant to start the diagnosis from (in ' + '%Y-%m-%dT%H:%M:%S.%fZ format).')
    parser.add_argument('--end-time', help='Time instant to stop the diagnosis at (in ' + '%Y-%m-%dT%H:%M:%S.%fZ format).')
    parser.add_argument('--job-id', hidden=True, help='The job on which to perform the diagnosis.', action=actions.DeprecationAction('--job-id', warn='The {flag_name} option is deprecated and will be removed in upcoming release; use --job-ids instead.', removed=False))
    parser.add_argument('--yarn-application-id', hidden=True, help='The yarn application on which to perform the diagnosis.', action=actions.DeprecationAction('--yarn-application-id', warn='The {flag_name} option is deprecated and will be removed in upcoming release; use --yarn-application-ids instead.', removed=False))
    parser.add_argument('--workers', hidden=True, help='A list of workers in the cluster to run the diagnostic script ' + 'on.')
    parser.add_argument('--job-ids', help='A list of jobs on which to perform the diagnosis.')
    parser.add_argument('--yarn-application-ids', help='A list of yarn applications on which to perform the diagnosis.')
    parser.add_argument('--tarball-gcs-dir', hidden=True, help='GCS Bucket location to store the results.')