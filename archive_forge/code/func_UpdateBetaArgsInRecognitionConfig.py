from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def UpdateBetaArgsInRecognitionConfig(self, args, config):
    """Updates config from command line arguments."""
    config.alternativeLanguageCodes = args.additional_language_codes
    if args.enable_speaker_diarization or args.min_diarization_speaker_count or args.max_diarization_speaker_count or args.diarization_speaker_count:
        speaker_config = config.diarizationConfig = config.field_by_name('diarizationConfig').message_type(enableSpeakerDiarization=True)
        if args.min_diarization_speaker_count:
            speaker_config.minSpeakerCount = args.min_diarization_speaker_count
        if args.max_diarization_speaker_count:
            speaker_config.maxSpeakerCount = args.max_diarization_speaker_count
        if args.diarization_speaker_count:
            if args.min_diarization_speaker_count or args.max_diarization_speaker_count:
                raise exceptions.InvalidArgumentException('--diarization-speaker-count', 'deprecated flag cannot be used with --max/min_diarization_speaker_count flags')
            speaker_config.minSpeakerCount = args.diarization_speaker_count
            speaker_config.maxSpeakerCount = args.diarization_speaker_count
    config.enableWordConfidence = args.include_word_confidence