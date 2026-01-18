from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def AddAllFlagsToParser(parser, create=False):
    """Parses all flags for v2 STT API."""
    AddRecognizerArgToParser(parser)
    base.ASYNC_FLAG.AddToParser(parser)
    base.ASYNC_FLAG.SetDefault(parser, False)
    parser.add_argument('--display-name', help='      Name of this recognizer as it appears in UIs.\n      ')
    parser.add_argument('--model', required=create, help='      `latest_long` or `latest_short`\n      ')
    parser.add_argument('--language-codes', metavar='LANGUAGE_CODE', required=create, type=arg_parsers.ArgList(), help='      Language code is one of `en-US`, `en-GB`, `fr-FR`.\n      Check [documentation](https://cloud.google.com/speech-to-text/docs/multiple-languages)\n      for using more than one language code.\n      ')
    parser.add_argument('--profanity-filter', type=arg_parsers.ArgBoolean(), help='      If true, the server will censor profanities.\n      ')
    parser.add_argument('--enable-word-time-offsets', type=arg_parsers.ArgBoolean(), help='      If true, the top result includes a list of words and their timestamps.\n      ')
    parser.add_argument('--enable-word-confidence', type=arg_parsers.ArgBoolean(), help='      If true, the top result includes a list of words and the confidence for\n      those words.\n      ')
    parser.add_argument('--enable-automatic-punctuation', type=arg_parsers.ArgBoolean(), help='      If true, adds punctuation to recognition result hypotheses.\n      ')
    parser.add_argument('--enable-spoken-punctuation', type=arg_parsers.ArgBoolean(), help='      If true, replaces spoken punctuation with the corresponding symbols in the request.\n      ')
    parser.add_argument('--enable-spoken-emojis', type=arg_parsers.ArgBoolean(), help='      If true, adds spoken emoji formatting.\n      ')
    parser.add_argument('--min-speaker-count', type=arg_parsers.BoundedInt(1, 6), help='        Minimum number of speakers in the conversation.\n        ')
    parser.add_argument('--max-speaker-count', type=arg_parsers.BoundedInt(1, 6), help='        Maximum number of speakers in the conversation.\n        ')
    parser.add_argument('--encoding', help='          Encoding format of the provided audio. For headerless formats, must be set to `LINEAR16`, `MULAW,` or `ALAW`. For other formats, set to `AUTO`.\n          ')
    parser.add_argument('--sample-rate', type=arg_parsers.BoundedInt(8000, 48000), help='          Sample rate in Hertz of the audio data sent for recognition.\n          Required if using explicit decoding.\n          ')
    parser.add_argument('--audio-channel-count', type=arg_parsers.BoundedInt(1, 8), help='          Number of channels present in the audio data sent for recognition.\n          Supported for LINEAR16, MULAW, ALAW.\n          ')