from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ml.speech import util
from googlecloudsdk.command_lib.util.apis import arg_utils
def GetRecordingDeviceTypeMapper(version):
    messages = apis.GetMessagesModule(util.SPEECH_API, version)
    return arg_utils.ChoiceEnumMapper('--recording-device-type', messages.RecognitionMetadata.RecordingDeviceTypeValueValuesEnum, action=MakeDeprecatedRecgonitionFlagAction('recording-device-type'), custom_mappings={'SMARTPHONE': ('smartphone', 'Speech was recorded on a smartphone.'), 'PC': ('pc', 'Speech was recorded using a personal computer or tablet.'), 'PHONE_LINE': ('phone-line', 'Speech was recorded over a phone line.'), 'VEHICLE': ('vehicle', 'Speech was recorded in a vehicle.'), 'OTHER_OUTDOOR_DEVICE': ('outdoor', 'Speech was recorded outdoors.'), 'OTHER_INDOOR_DEVICE': ('indoor', 'Speech was recorded indoors.')}, help_str='The device type through which the original audio was recorded on.', include_filter=lambda x: not x.endswith('UNSPECIFIED'))