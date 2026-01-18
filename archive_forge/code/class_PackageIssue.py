from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageIssue(_messages.Message):
    """A detail for a distro and package this vulnerability occurrence was
  found in and its associated fix (if one is available).

  Enums:
    EffectiveSeverityValueValuesEnum: Output only. The distro or language
      system assigned severity for this vulnerability when that is available
      and note provider assigned severity when it is not available.

  Fields:
    affectedCpeUri: Required. The [CPE
      URI](https://cpe.mitre.org/specification/) this vulnerability was found
      in.
    affectedPackage: Required. The package this vulnerability was found in.
    affectedVersion: Required. The version of the package that is installed on
      the resource affected by this vulnerability.
    effectiveSeverity: Output only. The distro or language system assigned
      severity for this vulnerability when that is available and note provider
      assigned severity when it is not available.
    fileLocation: The location at which this package was found.
    fixAvailable: Output only. Whether a fix is available for this package.
    fixedCpeUri: The [CPE URI](https://cpe.mitre.org/specification/) this
      vulnerability was fixed in. It is possible for this to be different from
      the affected_cpe_uri.
    fixedPackage: The package this vulnerability was fixed in. It is possible
      for this to be different from the affected_package.
    fixedVersion: Required. The version of the package this vulnerability was
      fixed in. Setting this to VersionKind.MAXIMUM means no fix is yet
      available.
    packageType: The type of package (e.g. OS, MAVEN, GO).
  """

    class EffectiveSeverityValueValuesEnum(_messages.Enum):
        """Output only. The distro or language system assigned severity for this
    vulnerability when that is available and note provider assigned severity
    when it is not available.

    Values:
      SEVERITY_UNSPECIFIED: Unknown.
      MINIMAL: Minimal severity.
      LOW: Low severity.
      MEDIUM: Medium severity.
      HIGH: High severity.
      CRITICAL: Critical severity.
    """
        SEVERITY_UNSPECIFIED = 0
        MINIMAL = 1
        LOW = 2
        MEDIUM = 3
        HIGH = 4
        CRITICAL = 5
    affectedCpeUri = _messages.StringField(1)
    affectedPackage = _messages.StringField(2)
    affectedVersion = _messages.MessageField('Version', 3)
    effectiveSeverity = _messages.EnumField('EffectiveSeverityValueValuesEnum', 4)
    fileLocation = _messages.MessageField('GrafeasV1FileLocation', 5, repeated=True)
    fixAvailable = _messages.BooleanField(6)
    fixedCpeUri = _messages.StringField(7)
    fixedPackage = _messages.StringField(8)
    fixedVersion = _messages.MessageField('Version', 9)
    packageType = _messages.StringField(10)