from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskExecutionSpec(_messages.Message):
    """Execution related settings, like retry and service_account.

  Messages:
    ArgsValue: Optional. The arguments to pass to the task. The args can use
      placeholders of the format ${placeholder} as part of key/value string.
      These will be interpolated before passing the args to the driver.
      Currently supported placeholders: - ${task_id} - ${job_time} To pass
      positional args, set the key as TASK_ARGS. The value should be a comma-
      separated string of all the positional arguments. To use a delimiter
      other than comma, refer to
      https://cloud.google.com/sdk/gcloud/reference/topic/escaping. In case of
      other keys being present in the args, then TASK_ARGS will be passed as
      the last argument.

  Fields:
    args: Optional. The arguments to pass to the task. The args can use
      placeholders of the format ${placeholder} as part of key/value string.
      These will be interpolated before passing the args to the driver.
      Currently supported placeholders: - ${task_id} - ${job_time} To pass
      positional args, set the key as TASK_ARGS. The value should be a comma-
      separated string of all the positional arguments. To use a delimiter
      other than comma, refer to
      https://cloud.google.com/sdk/gcloud/reference/topic/escaping. In case of
      other keys being present in the args, then TASK_ARGS will be passed as
      the last argument.
    kmsKey: Optional. The Cloud KMS key to use for encryption, of the form:
      projects/{project_number}/locations/{location_id}/keyRings/{key-ring-
      name}/cryptoKeys/{key-name}.
    maxJobExecutionLifetime: Optional. The maximum duration after which the
      job execution is expired.
    project: Optional. The project in which jobs are run. By default, the
      project containing the Lake is used. If a project is provided, the
      ExecutionSpec.service_account must belong to this project.
    serviceAccount: Required. Service account to use to execute a task. If not
      provided, the default Compute service account for the project is used.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ArgsValue(_messages.Message):
        """Optional. The arguments to pass to the task. The args can use
    placeholders of the format ${placeholder} as part of key/value string.
    These will be interpolated before passing the args to the driver.
    Currently supported placeholders: - ${task_id} - ${job_time} To pass
    positional args, set the key as TASK_ARGS. The value should be a comma-
    separated string of all the positional arguments. To use a delimiter other
    than comma, refer to
    https://cloud.google.com/sdk/gcloud/reference/topic/escaping. In case of
    other keys being present in the args, then TASK_ARGS will be passed as the
    last argument.

    Messages:
      AdditionalProperty: An additional property for a ArgsValue object.

    Fields:
      additionalProperties: Additional properties of type ArgsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ArgsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    args = _messages.MessageField('ArgsValue', 1)
    kmsKey = _messages.StringField(2)
    maxJobExecutionLifetime = _messages.StringField(3)
    project = _messages.StringField(4)
    serviceAccount = _messages.StringField(5)