from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Trial(_messages.Message):
    """A message representing a Trial. A Trial contains a unique set of
  Parameters that has been or will be evaluated, along with the objective
  metrics got by running the Trial.

  Enums:
    StateValueValuesEnum: Output only. The detailed state of the Trial.

  Messages:
    WebAccessUrisValue: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
      interactive-shell) (one URI for each training node). Only available if
      this trial is part of a HyperparameterTuningJob and the job's
      trial_job_spec.enable_web_access field is `true`. The keys are names of
      each node used for the trial; for example, `workerpool0-0` for the
      primary node, `workerpool1-0` for the first node in the second worker
      pool, and `workerpool1-1` for the second node in the second worker pool.
      The values are the URIs for each node's interactive shell.

  Fields:
    clientId: Output only. The identifier of the client that originally
      requested this Trial. Each client is identified by a unique client_id.
      When a client asks for a suggestion, Vertex AI Vizier will assign it a
      Trial. The client should evaluate the Trial, complete it, and report
      back to Vertex AI Vizier. If suggestion is asked again by same client_id
      before the Trial is completed, the same Trial will be returned. Multiple
      clients with different client_ids can ask for suggestions
      simultaneously, each of them will get their own Trial.
    customJob: Output only. The CustomJob name linked to the Trial. It's set
      for a HyperparameterTuningJob's Trial.
    endTime: Output only. Time when the Trial's status changed to `SUCCEEDED`
      or `INFEASIBLE`.
    finalMeasurement: Output only. The final measurement containing the
      objective value.
    id: Output only. The identifier of the Trial assigned by the service.
    infeasibleReason: Output only. A human readable string describing why the
      Trial is infeasible. This is set only if Trial state is `INFEASIBLE`.
    measurements: Output only. A list of measurements that are strictly
      lexicographically ordered by their induced tuples (steps,
      elapsed_duration). These are used for early stopping computations.
    name: Output only. Resource name of the Trial assigned by the service.
    parameters: Output only. The parameters of the Trial.
    startTime: Output only. Time when the Trial was started.
    state: Output only. The detailed state of the Trial.
    webAccessUris: Output only. URIs for accessing [interactive
      shells](https://cloud.google.com/vertex-ai/docs/training/monitor-debug-
      interactive-shell) (one URI for each training node). Only available if
      this trial is part of a HyperparameterTuningJob and the job's
      trial_job_spec.enable_web_access field is `true`. The keys are names of
      each node used for the trial; for example, `workerpool0-0` for the
      primary node, `workerpool1-0` for the first node in the second worker
      pool, and `workerpool1-1` for the second node in the second worker pool.
      The values are the URIs for each node's interactive shell.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The detailed state of the Trial.

    Values:
      STATE_UNSPECIFIED: The Trial state is unspecified.
      REQUESTED: Indicates that a specific Trial has been requested, but it
        has not yet been suggested by the service.
      ACTIVE: Indicates that the Trial has been suggested.
      STOPPING: Indicates that the Trial should stop according to the service.
      SUCCEEDED: Indicates that the Trial is completed successfully.
      INFEASIBLE: Indicates that the Trial should not be attempted again. The
        service will set a Trial to INFEASIBLE when it's done but missing the
        final_measurement.
    """
        STATE_UNSPECIFIED = 0
        REQUESTED = 1
        ACTIVE = 2
        STOPPING = 3
        SUCCEEDED = 4
        INFEASIBLE = 5

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
    clientId = _messages.StringField(1)
    customJob = _messages.StringField(2)
    endTime = _messages.StringField(3)
    finalMeasurement = _messages.MessageField('GoogleCloudAiplatformV1Measurement', 4)
    id = _messages.StringField(5)
    infeasibleReason = _messages.StringField(6)
    measurements = _messages.MessageField('GoogleCloudAiplatformV1Measurement', 7, repeated=True)
    name = _messages.StringField(8)
    parameters = _messages.MessageField('GoogleCloudAiplatformV1TrialParameter', 9, repeated=True)
    startTime = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    webAccessUris = _messages.MessageField('WebAccessUrisValue', 12)