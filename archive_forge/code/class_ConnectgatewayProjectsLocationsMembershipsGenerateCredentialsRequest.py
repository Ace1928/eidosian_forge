from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ConnectgatewayProjectsLocationsMembershipsGenerateCredentialsRequest(_messages.Message):
    """A ConnectgatewayProjectsLocationsMembershipsGenerateCredentialsRequest
  object.

  Fields:
    forceUseAgent: Optional. Whether to force the use of Connect Agent-based
      transport. This will return a configuration that uses Connect Agent as
      the underlying transport mechanism for cluster types that would
      otherwise have used a different transport. Requires that Connect Agent
      be installed on the cluster. Setting this field to false is equivalent
      to not setting it.
    kubernetesNamespace: Optional. The namespace to use in the kubeconfig
      context. If this field is specified, the server will set the `namespace`
      field in kubeconfig context. If not specified, the `namespace` field is
      omitted.
    name: Required. The Fleet membership resource.
    version: Optional. The Connect Gateway version to be used in the resulting
      configuration. Leave this field blank to let the server choose the
      version (recommended).
  """
    forceUseAgent = _messages.BooleanField(1)
    kubernetesNamespace = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    version = _messages.StringField(4)