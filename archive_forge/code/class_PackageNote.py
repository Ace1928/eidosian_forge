from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageNote(_messages.Message):
    """PackageNote represents a particular package version.

  Enums:
    ArchitectureValueValuesEnum: The CPU architecture for which packages in
      this distribution channel were built. Architecture will be blank for
      language packages.

  Fields:
    architecture: The CPU architecture for which packages in this distribution
      channel were built. Architecture will be blank for language packages.
    cpeUri: The cpe_uri in [CPE format](https://cpe.mitre.org/specification/)
      denoting the package manager version distributing a package. The cpe_uri
      will be blank for language packages.
    description: The description of this package.
    digest: Hash value, typically a file digest, that allows unique
      identification a specific package.
    distribution: Deprecated. The various channels by which a package is
      distributed.
    license: Licenses that have been declared by the authors of the package.
    maintainer: A freeform text denoting the maintainer of this package.
    name: Required. Immutable. The name of the package.
    packageType: The type of package; whether native or non native (e.g., ruby
      gems, node.js packages, etc.).
    url: The homepage for this package.
    version: The version of the package.
  """

    class ArchitectureValueValuesEnum(_messages.Enum):
        """The CPU architecture for which packages in this distribution channel
    were built. Architecture will be blank for language packages.

    Values:
      ARCHITECTURE_UNSPECIFIED: Unknown architecture.
      X86: X86 architecture.
      X64: X64 architecture.
    """
        ARCHITECTURE_UNSPECIFIED = 0
        X86 = 1
        X64 = 2
    architecture = _messages.EnumField('ArchitectureValueValuesEnum', 1)
    cpeUri = _messages.StringField(2)
    description = _messages.StringField(3)
    digest = _messages.MessageField('Digest', 4, repeated=True)
    distribution = _messages.MessageField('Distribution', 5, repeated=True)
    license = _messages.MessageField('License', 6)
    maintainer = _messages.StringField(7)
    name = _messages.StringField(8)
    packageType = _messages.StringField(9)
    url = _messages.StringField(10)
    version = _messages.MessageField('Version', 11)