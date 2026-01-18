from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def MakeRecognitionConfig(self, args, messages):
    """Make RecognitionConfig message from given arguments."""
    config = messages.RecognitionConfig(languageCode=args.language_code if args.language_code else args.language, encoding=self._encoding_type_mapper.GetEnumForChoice(args.encoding.replace('_', '-').lower()), sampleRateHertz=args.sample_rate, audioChannelCount=args.audio_channel_count, maxAlternatives=args.max_alternatives, enableWordTimeOffsets=args.include_word_time_offsets, enableSeparateRecognitionPerChannel=args.separate_channel_recognition, profanityFilter=args.filter_profanity, speechContexts=[messages.SpeechContext(phrases=args.hints)])
    if args.enable_automatic_punctuation:
        config.enableAutomaticPunctuation = args.enable_automatic_punctuation
    if args.model is not None:
        if args.model in ['default', 'command_and_search', 'phone_call', 'latest_long', 'latest_short', 'medical_conversation', 'medical_dictation', 'telephony', 'telephony_short']:
            config.model = args.model
        elif args.model == 'phone_call_enhanced':
            config.model = 'phone_call'
            config.useEnhanced = True
        elif args.model == 'video_enhanced':
            config.model = 'video'
            config.useEnhanced = True
    return config