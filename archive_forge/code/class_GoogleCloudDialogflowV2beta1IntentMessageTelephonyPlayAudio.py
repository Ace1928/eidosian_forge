from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1IntentMessageTelephonyPlayAudio(_messages.Message):
    """Plays audio from a file in Telephony Gateway.

  Fields:
    audioUri: Required. URI to a Google Cloud Storage object containing the
      audio to play, e.g., "gs://bucket/object". The object must contain a
      single channel (mono) of linear PCM audio (2 bytes / sample) at 8kHz.
      This object must be readable by the `service-@gcp-sa-
      dialogflow.iam.gserviceaccount.com` service account where is the number
      of the Telephony Gateway project (usually the same as the Dialogflow
      agent project). If the Google Cloud Storage bucket is in the Telephony
      Gateway project, this permission is added by default when enabling the
      Dialogflow V2 API. For audio from other sources, consider using the
      `TelephonySynthesizeSpeech` message with SSML.
  """
    audioUri = _messages.StringField(1)