from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class WebAccessUrisValue(_messages.Message):
    """Output only. URIs for accessing [interactive
    shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
    interactive-shell) (one URI for each training node). Only available if
    this trial is part of a HyperparameterTuningJob and the job's
    trial_job_spec.enable_web_access field is `true`. The keys are names of
    each node used for the trial; for example, `workerpool0-0` for the primary
    node, `workerpool1-0` for the first node in the second worker pool, and
    `workerpool1-1` for the second node in the second worker pool. The values
    are the URIs for each node's interactive shell.

    Messages:
      AdditionalProperty: An additional property for a WebAccessUrisValue
        object.

    Fields:
      additionalProperties: Additional properties of type WebAccessUrisValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a WebAccessUrisValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)