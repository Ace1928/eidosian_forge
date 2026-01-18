from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BundlesValue(_messages.Message):
    """map of bundle name to BundleInstallSpec. The bundle name maps to the
    `bundleName` key in the `policycontroller.gke.io/constraintData`
    annotation on a constraint.

    Messages:
      AdditionalProperty: An additional property for a BundlesValue object.

    Fields:
      additionalProperties: Additional properties of type BundlesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BundlesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerBundleInstallSpec attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('PolicyControllerBundleInstallSpec', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)