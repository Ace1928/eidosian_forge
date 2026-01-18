from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddBetaRecognizeArgsToParser(self, parser):
    """Add beta arguments."""
    parser.add_argument('--additional-language-codes', type=arg_parsers.ArgList(), default=[], metavar='LANGUAGE_CODE', help='The BCP-47 language tags of other languages that the speech may be in.\nUp to 3 can be provided.\n\nIf alternative languages are listed, recognition result will contain recognition\nin the most likely language detected including the main language-code.')
    speaker_args = parser.add_group(required=False)
    speaker_args.add_argument('--diarization-speaker-count', type=int, hidden=True, action=actions.DeprecationAction('--diarization-speaker-count', warn='The `--diarization-speaker-count` flag is deprecated. Use the `--min-diarization-speaker-count` and/or `--max-diarization-speaker-count` flag instead.'), help='Estimated number of speakers in the conversation being recognized.')
    speaker_args.add_argument('--min-diarization-speaker-count', type=int, help='Minimum estimated number of speakers in the conversation being recognized.')
    speaker_args.add_argument('--max-diarization-speaker-count', type=int, help='Maximum estimated number of speakers in the conversation being recognized.')
    speaker_args.add_argument('--enable-speaker-diarization', action='store_true', required=True, help='Enable speaker detection for each recognized word in the top alternative of the recognition result using an integer speaker_tag provided in the WordInfo.')
    parser.add_argument('--include-word-confidence', action='store_true', help='Include a list of words and the confidence for those words in the top result.')