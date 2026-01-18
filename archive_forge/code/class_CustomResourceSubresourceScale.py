from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CustomResourceSubresourceScale(_messages.Message):
    """CustomResourceSubresourceScale defines how to serve the scale
  subresource for CustomResources.

  Fields:
    labelSelectorPath: LabelSelectorPath defines the JSON path inside of a
      CustomResource that corresponds to Scale.Status.Selector. Only JSON
      paths without the array notation are allowed. Must be a JSON Path under
      .status. Must be set to work with HPA. If there is no value under the
      given path in the CustomResource, the status label selector value in the
      /scale subresource will default to the empty string. +optional
    specReplicasPath: SpecReplicasPath defines the JSON path inside of a
      CustomResource that corresponds to Scale.Spec.Replicas. Only JSON paths
      without the array notation are allowed. Must be a JSON Path under .spec.
      If there is no value under the given path in the CustomResource, the
      /scale subresource will return an error on GET.
    statusReplicasPath: StatusReplicasPath defines the JSON path inside of a
      CustomResource that corresponds to Scale.Status.Replicas. Only JSON
      paths without the array notation are allowed. Must be a JSON Path under
      .status. If there is no value under the given path in the
      CustomResource, the status replica value in the /scale subresource will
      default to 0.
  """
    labelSelectorPath = _messages.StringField(1)
    specReplicasPath = _messages.StringField(2)
    statusReplicasPath = _messages.StringField(3)