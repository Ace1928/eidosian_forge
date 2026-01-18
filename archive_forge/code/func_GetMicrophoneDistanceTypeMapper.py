from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetMicrophoneDistanceTypeMapper(version):
    messages = apis.GetMessagesModule(util.SPEECH_API, version)
    return arg_utils.ChoiceEnumMapper('--microphone-distance', messages.RecognitionMetadata.MicrophoneDistanceValueValuesEnum, action=MakeDeprecatedRecgonitionFlagAction('microphone-distance'), custom_mappings={'NEARFIELD': ('nearfield', 'The audio was captured from a microphone close to the speaker, generally within\n 1 meter. Examples include a phone, dictaphone, or handheld microphone.'), 'MIDFIELD': ('midfield', 'The speaker is within 3 meters of the microphone.'), 'FARFIELD': ('farfield', 'The speaker is more than 3 meters away from the microphone.')}, help_str='The distance at which the audio device is placed to record the conversation.', include_filter=lambda x: not x.endswith('UNSPECIFIED'))