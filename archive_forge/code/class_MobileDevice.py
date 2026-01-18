from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MobileDevice(_messages.Message):
    """JSON template for Mobile Device resource in Directory API.

  Messages:
    ApplicationsValueListEntry: A ApplicationsValueListEntry object.

  Fields:
    adbStatus: Adb (USB debugging) enabled or disabled on device (Read-only)
    applications: List of applications installed on Mobile Device
    basebandVersion: Mobile Device Baseband version (Read-only)
    bootloaderVersion: Mobile Device Bootloader version (Read-only)
    brand: Mobile Device Brand (Read-only)
    buildNumber: Mobile Device Build number (Read-only)
    defaultLanguage: The default locale used on the Mobile Device (Read-only)
    developerOptionsStatus: Developer options enabled or disabled on device
      (Read-only)
    deviceCompromisedStatus: Mobile Device compromised status (Read-only)
    deviceId: Mobile Device serial number (Read-only)
    devicePasswordStatus: DevicePasswordStatus (Read-only)
    email: List of owner user's email addresses (Read-only)
    encryptionStatus: Mobile Device Encryption Status (Read-only)
    etag: ETag of the resource.
    firstSync: Date and time the device was first synchronized with the policy
      settings in the G Suite administrator control panel (Read-only)
    hardware: Mobile Device Hardware (Read-only)
    hardwareId: Mobile Device Hardware Id (Read-only)
    imei: Mobile Device IMEI number (Read-only)
    kernelVersion: Mobile Device Kernel version (Read-only)
    kind: Kind of resource this is.
    lastSync: Date and time the device was last synchronized with the policy
      settings in the G Suite administrator control panel (Read-only)
    managedAccountIsOnOwnerProfile: Boolean indicating if this account is on
      owner/primary profile or not (Read-only)
    manufacturer: Mobile Device manufacturer (Read-only)
    meid: Mobile Device MEID number (Read-only)
    model: Name of the model of the device
    name: List of owner user's names (Read-only)
    networkOperator: Mobile Device mobile or network operator (if available)
      (Read-only)
    os: Name of the mobile operating system
    otherAccountsInfo: List of accounts added on device (Read-only)
    privilege: DMAgentPermission (Read-only)
    releaseVersion: Mobile Device release version version (Read-only)
    resourceId: Unique identifier of Mobile Device (Read-only)
    securityPatchLevel: Mobile Device Security patch level (Read-only)
    serialNumber: Mobile Device SSN or Serial Number (Read-only)
    status: Status of the device (Read-only)
    supportsWorkProfile: Work profile supported on device (Read-only)
    type: The type of device (Read-only)
    unknownSourcesStatus: Unknown sources enabled or disabled on device (Read-
      only)
    userAgent: Mobile Device user agent
    wifiMacAddress: Mobile Device WiFi MAC address (Read-only)
  """

    class ApplicationsValueListEntry(_messages.Message):
        """A ApplicationsValueListEntry object.

    Fields:
      displayName: Display name of application
      packageName: Package name of application
      permission: List of Permissions for application
      versionCode: Version code of application
      versionName: Version name of application
    """
        displayName = _messages.StringField(1)
        packageName = _messages.StringField(2)
        permission = _messages.StringField(3, repeated=True)
        versionCode = _messages.IntegerField(4, variant=_messages.Variant.INT32)
        versionName = _messages.StringField(5)
    adbStatus = _messages.BooleanField(1)
    applications = _messages.MessageField('ApplicationsValueListEntry', 2, repeated=True)
    basebandVersion = _messages.StringField(3)
    bootloaderVersion = _messages.StringField(4)
    brand = _messages.StringField(5)
    buildNumber = _messages.StringField(6)
    defaultLanguage = _messages.StringField(7)
    developerOptionsStatus = _messages.BooleanField(8)
    deviceCompromisedStatus = _messages.StringField(9)
    deviceId = _messages.StringField(10)
    devicePasswordStatus = _messages.StringField(11)
    email = _messages.StringField(12, repeated=True)
    encryptionStatus = _messages.StringField(13)
    etag = _messages.StringField(14)
    firstSync = _message_types.DateTimeField(15)
    hardware = _messages.StringField(16)
    hardwareId = _messages.StringField(17)
    imei = _messages.StringField(18)
    kernelVersion = _messages.StringField(19)
    kind = _messages.StringField(20, default=u'admin#directory#mobiledevice')
    lastSync = _message_types.DateTimeField(21)
    managedAccountIsOnOwnerProfile = _messages.BooleanField(22)
    manufacturer = _messages.StringField(23)
    meid = _messages.StringField(24)
    model = _messages.StringField(25)
    name = _messages.StringField(26, repeated=True)
    networkOperator = _messages.StringField(27)
    os = _messages.StringField(28)
    otherAccountsInfo = _messages.StringField(29, repeated=True)
    privilege = _messages.StringField(30)
    releaseVersion = _messages.StringField(31)
    resourceId = _messages.StringField(32)
    securityPatchLevel = _messages.IntegerField(33)
    serialNumber = _messages.StringField(34)
    status = _messages.StringField(35)
    supportsWorkProfile = _messages.BooleanField(36)
    type = _messages.StringField(37)
    unknownSourcesStatus = _messages.BooleanField(38)
    userAgent = _messages.StringField(39)
    wifiMacAddress = _messages.StringField(40)