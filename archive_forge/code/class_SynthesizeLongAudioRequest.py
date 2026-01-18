from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts
class SynthesizeLongAudioRequest(proto.Message):
    """The top-level message sent by the client for the
    ``SynthesizeLongAudio`` method.

    Attributes:
        parent (str):
            The resource states of the request in the form of
            ``projects/*/locations/*``.
        input (google.cloud.texttospeech_v1.types.SynthesisInput):
            Required. The Synthesizer requires either
            plain text or SSML as input. While Long Audio is
            in preview, SSML is temporarily unsupported.
        audio_config (google.cloud.texttospeech_v1.types.AudioConfig):
            Required. The configuration of the
            synthesized audio.
        output_gcs_uri (str):
            Required. Specifies a Cloud Storage URI for the synthesis
            results. Must be specified in the format:
            ``gs://bucket_name/object_name``, and the bucket must
            already exist.
        voice (google.cloud.texttospeech_v1.types.VoiceSelectionParams):
            Required. The desired voice of the
            synthesized audio.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    input: cloud_tts.SynthesisInput = proto.Field(proto.MESSAGE, number=2, message=cloud_tts.SynthesisInput)
    audio_config: cloud_tts.AudioConfig = proto.Field(proto.MESSAGE, number=3, message=cloud_tts.AudioConfig)
    output_gcs_uri: str = proto.Field(proto.STRING, number=4)
    voice: cloud_tts.VoiceSelectionParams = proto.Field(proto.MESSAGE, number=5, message=cloud_tts.VoiceSelectionParams)