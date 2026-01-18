from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetEncodingTypeMapper(version):
    messages = apis.GetMessagesModule(util.SPEECH_API, version)
    return arg_utils.ChoiceEnumMapper('--encoding', messages.RecognitionConfig.EncodingValueValuesEnum, default='encoding-unspecified', help_str='The type of encoding of the file. Required if the file format is not WAV or FLAC.')