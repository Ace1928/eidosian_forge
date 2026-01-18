from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AndroidModel(_messages.Message):
    """A description of an Android device tests may be run on.

  Enums:
    FormValueValuesEnum: Whether this device is virtual or physical.
    FormFactorValueValuesEnum: Whether this device is a phone, tablet,
      wearable, etc.

  Fields:
    brand: The company that this device is branded with. Example: "Google",
      "Samsung".
    codename: The name of the industrial design. This corresponds to
      android.os.Build.DEVICE.
    form: Whether this device is virtual or physical.
    formFactor: Whether this device is a phone, tablet, wearable, etc.
    id: The unique opaque id for this model. Use this for invoking the
      TestExecutionService.
    lowFpsVideoRecording: True if and only if tests with this model are
      recorded by stitching together screenshots. See
      use_low_spec_video_recording in device config.
    manufacturer: The manufacturer of this device.
    name: The human-readable marketing name for this device model. Examples:
      "Nexus 5", "Galaxy S5".
    perVersionInfo: Version-specific information of an Android model.
    screenDensity: Screen density in DPI. This corresponds to
      ro.sf.lcd_density
    screenX: Screen size in the horizontal (X) dimension measured in pixels.
    screenY: Screen size in the vertical (Y) dimension measured in pixels.
    supportedAbis: The list of supported ABIs for this device. This
      corresponds to either android.os.Build.SUPPORTED_ABIS (for API level 21
      and above) or android.os.Build.CPU_ABI/CPU_ABI2. The most preferred ABI
      is the first element in the list. Elements are optionally prefixed by
      "version_id:" (where version_id is the id of an AndroidVersion),
      denoting an ABI that is supported only on a particular version.
    supportedVersionIds: The set of Android versions this device supports.
    tags: Tags for this dimension. Examples: "default", "preview",
      "deprecated".
    thumbnailUrl: URL of a thumbnail image (photo) of the device.
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

    class FormValueValuesEnum(_messages.Enum):
        """Whether this device is virtual or physical.

    Values:
      DEVICE_FORM_UNSPECIFIED: Do not use. For proto versioning only.
      VIRTUAL: Android virtual device using Compute Engine native
        virtualization. Firebase Test Lab only.
      PHYSICAL: Actual hardware.
      EMULATOR: Android virtual device using emulator in nested
        virtualization. Equivalent to Android Studio.
    """
        DEVICE_FORM_UNSPECIFIED = 0
        VIRTUAL = 1
        PHYSICAL = 2
        EMULATOR = 3
    brand = _messages.StringField(1)
    codename = _messages.StringField(2)
    form = _messages.EnumField('FormValueValuesEnum', 3)
    formFactor = _messages.EnumField('FormFactorValueValuesEnum', 4)
    id = _messages.StringField(5)
    lowFpsVideoRecording = _messages.BooleanField(6)
    manufacturer = _messages.StringField(7)
    name = _messages.StringField(8)
    perVersionInfo = _messages.MessageField('PerAndroidVersionInfo', 9, repeated=True)
    screenDensity = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    screenX = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    screenY = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    supportedAbis = _messages.StringField(13, repeated=True)
    supportedVersionIds = _messages.StringField(14, repeated=True)
    tags = _messages.StringField(15, repeated=True)
    thumbnailUrl = _messages.StringField(16)