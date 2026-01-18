from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerPolicyContentState(_messages.Message):
    """The state of the policy controller policy content

  Messages:
    BundleStatesValue: The state of the any bundles included in the chosen
      version of the manifest

  Fields:
    bundleStates: The state of the any bundles included in the chosen version
      of the manifest
    referentialSyncConfigState: The state of the referential data sync
      configuration. This could represent the state of either the syncSet
      object(s) or the config object, depending on the version of PoCo
      configured by the user.
    templateLibraryState: The state of the template library
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class BundleStatesValue(_messages.Message):
        """The state of the any bundles included in the chosen version of the
    manifest

    Messages:
      AdditionalProperty: An additional property for a BundleStatesValue
        object.

    Fields:
      additionalProperties: Additional properties of type BundleStatesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a BundleStatesValue object.

      Fields:
        key: Name of the additional property.
        value: A PolicyControllerOnClusterState attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('PolicyControllerOnClusterState', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    bundleStates = _messages.MessageField('BundleStatesValue', 1)
    referentialSyncConfigState = _messages.MessageField('PolicyControllerOnClusterState', 2)
    templateLibraryState = _messages.MessageField('PolicyControllerOnClusterState', 3)