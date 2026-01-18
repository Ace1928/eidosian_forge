from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MethodSettings(_messages.Message):
    """Describes the generator configuration for a method.

  Fields:
    autoPopulatedFields: List of top-level fields of the request message, that
      should be automatically populated by the client libraries based on their
      (google.api.field_info).format. Currently supported format: UUID4.
      Example of a YAML configuration: publishing: method_settings: -
      selector: google.example.v1.ExampleService.CreateExample
      auto_populated_fields: - request_id
    longRunning: Describes settings to use for long-running operations when
      generating API methods for RPCs. Complements RPCs that use the
      annotations in google/longrunning/operations.proto. Example of a YAML
      configuration:: publishing: method_settings: - selector:
      google.cloud.speech.v2.Speech.BatchRecognize long_running:
      initial_poll_delay: seconds: 60 # 1 minute poll_delay_multiplier: 1.5
      max_poll_delay: seconds: 360 # 6 minutes total_poll_timeout: seconds:
      54000 # 90 minutes
    selector: The fully qualified name of the method, for which the options
      below apply. This is used to find the method to apply the options.
  """
    autoPopulatedFields = _messages.StringField(1, repeated=True)
    longRunning = _messages.MessageField('LongRunning', 2)
    selector = _messages.StringField(3)