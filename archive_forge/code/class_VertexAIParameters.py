from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VertexAIParameters(_messages.Message):
    """Parameters used in Vertex AI JobType executions.

  Messages:
    EnvValue: Environment variables. At most 100 environment variables can be
      specified and unique. Example: `GCP_BUCKET=gs://my-bucket/samples/`

  Fields:
    env: Environment variables. At most 100 environment variables can be
      specified and unique. Example: `GCP_BUCKET=gs://my-bucket/samples/`
    network: The full name of the Compute Engine
      [network](https://cloud.google.com/compute/docs/networks-and-
      firewalls#networks) to which the Job should be peered. For example,
      `projects/12345/global/networks/myVPC`. [Format](https://cloud.google.co
      m/compute/docs/reference/rest/v1/networks/insert) is of the form
      `projects/{project}/global/networks/{network}`. Where `{project}` is a
      project number, as in `12345`, and `{network}` is a network name.
      Private services access must already be configured for the network. If
      left unspecified, the job is not peered with any network.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class EnvValue(_messages.Message):
        """Environment variables. At most 100 environment variables can be
    specified and unique. Example: `GCP_BUCKET=gs://my-bucket/samples/`

    Messages:
      AdditionalProperty: An additional property for a EnvValue object.

    Fields:
      additionalProperties: Additional properties of type EnvValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a EnvValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    env = _messages.MessageField('EnvValue', 1)
    network = _messages.StringField(2)