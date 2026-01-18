from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IosModel(_messages.Message):
    """A description of an iOS device tests may be run on.

  Enums:
    FormFactorValueValuesEnum: Whether this device is a phone, tablet,
      wearable, etc.

  Fields:
    deviceCapabilities: Device capabilities. Copied from https://developer.app
      le.com/library/archive/documentation/DeviceInformation/Reference/iOSDevi
      ceCompatibility/DeviceCompatibilityMatrix/DeviceCompatibilityMatrix.html
    formFactor: Whether this device is a phone, tablet, wearable, etc.
    id: The unique opaque id for this model. Use this for invoking the
      TestExecutionService.
    name: The human-readable name for this device model. Examples: "iPhone
      4s", "iPad Mini 2".
    perVersionInfo: Version-specific information of an iOS model.
    screenDensity: Screen density in DPI.
    screenX: Screen size in the horizontal (X) dimension measured in pixels.
    screenY: Screen size in the vertical (Y) dimension measured in pixels.
    supportedVersionIds: The set of iOS major software versions this device
      supports.
    tags: Tags for this dimension. Examples: "default", "preview",
      "deprecated".
  """

    class FormFactorValueValuesEnum(_messages.Enum):
        """Whether this device is a phone, tablet, wearable, etc.

    Values:
      DEVICE_FORM_FACTOR_UNSPECIFIED: Do not use. For proto versioning only.
      PHONE: This device has the shape of a phone.
      TABLET: This device has the shape of a tablet.
      WEARABLE: This device has the shape of a watch or other wearable.
    """
        DEVICE_FORM_FACTOR_UNSPECIFIED = 0
        PHONE = 1
        TABLET = 2
        WEARABLE = 3
    deviceCapabilities = _messages.StringField(1, repeated=True)
    formFactor = _messages.EnumField('FormFactorValueValuesEnum', 2)
    id = _messages.StringField(3)
    name = _messages.StringField(4)
    perVersionInfo = _messages.MessageField('PerIosVersionInfo', 5, repeated=True)
    screenDensity = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    screenX = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    screenY = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    supportedVersionIds = _messages.StringField(9, repeated=True)
    tags = _messages.StringField(10, repeated=True)