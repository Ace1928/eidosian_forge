from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.command_lib.ai.hp_tuning_jobs import hp_tuning_jobs_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddCreateHpTuningJobFlags(parser, algorithm_enum):
    """Adds arguments for creating hp tuning job."""
    _HPTUNING_JOB_DISPLAY_NAME.AddToParser(parser)
    _HPTUNING_JOB_CONFIG.AddToParser(parser)
    _HPTUNING_MAX_TRIAL_COUNT.AddToParser(parser)
    _HPTUNING_PARALLEL_TRIAL_COUNT.AddToParser(parser)
    labels_util.AddCreateLabelsFlags(parser)
    flags.AddRegionResourceArg(parser, 'to create a hyperparameter tuning job', prompt_func=region_util.GetPromptForRegionFunc(constants.SUPPORTED_TRAINING_REGIONS))
    flags.TRAINING_SERVICE_ACCOUNT.AddToParser(parser)
    flags.NETWORK.AddToParser(parser)
    flags.ENABLE_WEB_ACCESS.AddToParser(parser)
    flags.ENABLE_DASHBOARD_ACCESS.AddToParser(parser)
    flags.AddKmsKeyResourceArg(parser, 'hyperparameter tuning job')
    arg_utils.ChoiceEnumMapper('--algorithm', algorithm_enum, help_str='Search algorithm specified for the given study. ').choice_arg.AddToParser(parser)