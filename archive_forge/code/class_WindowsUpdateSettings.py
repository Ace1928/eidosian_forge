from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WindowsUpdateSettings(_messages.Message):
    """Windows patching is performed using the Windows Update Agent.

  Enums:
    ClassificationsValueListEntryValuesEnum:

  Fields:
    classifications: Only apply updates of these windows update
      classifications. If empty, all updates are applied.
    excludes: List of KBs to exclude from update.
    exclusivePatches: An exclusive list of kbs to be updated. These are the
      only patches that will be updated. This field must not be used with
      other patch configurations.
  """

    class ClassificationsValueListEntryValuesEnum(_messages.Enum):
        """ClassificationsValueListEntryValuesEnum enum type.

    Values:
      CLASSIFICATION_UNSPECIFIED: Invalid. If classifications are included,
        they must be specified.
      CRITICAL: "A widely released fix for a specific problem that addresses a
        critical, non-security-related bug." [1]
      SECURITY: "A widely released fix for a product-specific, security-
        related vulnerability. Security vulnerabilities are rated by their
        severity. The severity rating is indicated in the Microsoft security
        bulletin as critical, important, moderate, or low." [1]
      DEFINITION: "A widely released and frequent software update that
        contains additions to a product's definition database. Definition
        databases are often used to detect objects that have specific
        attributes, such as malicious code, phishing websites, or junk mail."
        [1]
      DRIVER: "Software that controls the input and output of a device." [1]
      FEATURE_PACK: "New product functionality that is first distributed
        outside the context of a product release and that is typically
        included in the next full product release." [1]
      SERVICE_PACK: "A tested, cumulative set of all hotfixes, security
        updates, critical updates, and updates. Additionally, service packs
        may contain additional fixes for problems that are found internally
        since the release of the product. Service packs my also contain a
        limited number of customer-requested design changes or features." [1]
      TOOL: "A utility or feature that helps complete a task or set of tasks."
        [1]
      UPDATE_ROLLUP: "A tested, cumulative set of hotfixes, security updates,
        critical updates, and updates that are packaged together for easy
        deployment. A rollup generally targets a specific area, such as
        security, or a component of a product, such as Internet Information
        Services (IIS)." [1]
      UPDATE: "A widely released fix for a specific problem. An update
        addresses a noncritical, non-security-related bug." [1]
    """
        CLASSIFICATION_UNSPECIFIED = 0
        CRITICAL = 1
        SECURITY = 2
        DEFINITION = 3
        DRIVER = 4
        FEATURE_PACK = 5
        SERVICE_PACK = 6
        TOOL = 7
        UPDATE_ROLLUP = 8
        UPDATE = 9
    classifications = _messages.EnumField('ClassificationsValueListEntryValuesEnum', 1, repeated=True)
    excludes = _messages.StringField(2, repeated=True)
    exclusivePatches = _messages.StringField(3, repeated=True)