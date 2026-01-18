from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.ml_engine import flags
from googlecloudsdk.command_lib.ml_engine import jobs_util
from googlecloudsdk.command_lib.util.args import labels_util
def _AddSubmitTrainingArgs(parser):
    """Add arguments for `jobs submit training` command."""
    flags.JOB_NAME.AddToParser(parser)
    flags.PACKAGE_PATH.AddToParser(parser)
    flags.PACKAGES.AddToParser(parser)
    flags.GetModuleNameFlag(required=False).AddToParser(parser)
    compute_flags.AddRegionFlag(parser, 'machine learning training job', 'submit')
    flags.CONFIG.AddToParser(parser)
    flags.STAGING_BUCKET.AddToParser(parser)
    flags.GetJobDirFlag(upload_help=True).AddToParser(parser)
    flags.GetUserArgs(local=False).AddToParser(parser)
    jobs_util.ScaleTierFlagMap().choice_arg.AddToParser(parser)
    flags.RUNTIME_VERSION.AddToParser(parser)
    flags.AddPythonVersionFlag(parser, 'during training')
    flags.TRAINING_SERVICE_ACCOUNT.AddToParser(parser)
    flags.ENABLE_WEB_ACCESS.AddToParser(parser)
    sync_group = parser.add_mutually_exclusive_group()
    sync_group.add_argument('--async', action='store_true', dest='async_', help='(DEPRECATED) Display information about the operation in progress without waiting for the operation to complete. Enabled by default and can be omitted; use `--stream-logs` to run synchronously.')
    sync_group.add_argument('--stream-logs', action='store_true', help='Block until job completion and stream the logs while the job runs.\n\nNote that even if command execution is halted, the job will still run until cancelled with\n\n    $ gcloud ai-platform jobs cancel JOB_ID')
    labels_util.AddCreateLabelsFlags(parser)