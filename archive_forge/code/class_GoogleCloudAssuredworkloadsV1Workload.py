from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1Workload(_messages.Message):
    """A Workload object for managing highly regulated workloads of cloud
  customers.

  Enums:
    ComplianceRegimeValueValuesEnum: Required. Immutable. Compliance Regime
      associated with this workload.
    KajEnrollmentStateValueValuesEnum: Output only. Represents the KAJ
      enrollment state of the given workload.
    PartnerValueValuesEnum: Optional. Partner regime associated with this
      workload.

  Messages:
    LabelsValue: Optional. Labels applied to the workload.

  Fields:
    billingAccount: Optional. The billing account used for the resources which
      are direct children of workload. This billing account is initially
      associated with the resources created as part of Workload creation.
      After the initial creation of these resources, the customer can change
      the assigned billing account. The resource name has the form
      `billingAccounts/{billing_account_id}`. For example,
      `billingAccounts/012345-567890-ABCDEF`.
    complianceRegime: Required. Immutable. Compliance Regime associated with
      this workload.
    complianceStatus: Output only. Count of active Violations in the Workload.
    compliantButDisallowedServices: Output only. Urls for services which are
      compliant for this Assured Workload, but which are currently disallowed
      by the ResourceUsageRestriction org policy. Invoke
      RestrictAllowedResources endpoint to allow your project developers to
      use these services in their environment.
    createTime: Output only. Immutable. The Workload creation timestamp.
    displayName: Required. The user-assigned display name of the Workload.
      When present it must be between 4 to 30 characters. Allowed characters
      are: lowercase and uppercase letters, numbers, hyphen, and spaces.
      Example: My Workload
    ekmProvisioningResponse: Output only. Represents the Ekm Provisioning
      State of the given workload.
    enableSovereignControls: Optional. Indicates the sovereignty status of the
      given workload. Currently meant to be used by Europe/Canada customers.
    etag: Optional. ETag of the workload, it is calculated on the basis of the
      Workload contents. It will be used in Update & Delete operations.
    kajEnrollmentState: Output only. Represents the KAJ enrollment state of
      the given workload.
    kmsSettings: Input only. Settings used to create a CMEK crypto key. When
      set, a project with a KMS CMEK key is provisioned. This field is
      deprecated as of Feb 28, 2022. In order to create a Keyring, callers
      should specify, ENCRYPTION_KEYS_PROJECT or KEYRING in
      ResourceSettings.resource_type field.
    labels: Optional. Labels applied to the workload.
    name: Optional. The resource name of the workload. Format:
      organizations/{organization}/locations/{location}/workloads/{workload}
      Read-only.
    partner: Optional. Partner regime associated with this workload.
    partnerPermissions: Optional. Permissions granted to the AW Partner SA
      account for the customer workload
    provisionedResourcesParent: Input only. The parent resource for the
      resources managed by this Assured Workload. May be either empty or a
      folder resource which is a child of the Workload parent. If not
      specified all resources are created under the parent organization.
      Format: folders/{folder_id}
    resourceMonitoringEnabled: Output only. Indicates whether resource
      monitoring is enabled for workload or not. It is true when Resource feed
      is subscribed to AWM topic and AWM Service Agent Role is binded to AW
      Service Account for resource Assured workload.
    resourceSettings: Input only. Resource properties that are used to
      customize workload resources. These properties (such as custom project
      id) will be used to create workload resources if possible. This field is
      optional.
    resources: Output only. The resources associated with this workload. These
      resources will be created when creating the workload. If any of the
      projects already exist, the workload creation will fail. Always read
      only.
    saaEnrollmentResponse: Output only. Represents the SAA enrollment response
      of the given workload. SAA enrollment response is queried during
      GetWorkload call. In failure cases, user friendly error message is shown
      in SAA details page.
    violationNotificationsEnabled: Optional. Indicates whether the e-mail
      notification for a violation is enabled for a workload. This value will
      be by default True, and if not present will be considered as true. This
      should only be updated via updateWorkload call. Any Changes to this
      field during the createWorkload call will not be honored. This will
      always be true while creating the workload.
  """

    class ComplianceRegimeValueValuesEnum(_messages.Enum):
        """Required. Immutable. Compliance Regime associated with this workload.

    Values:
      COMPLIANCE_REGIME_UNSPECIFIED: Unknown compliance regime.
      IL4: Information protection as per DoD IL4 requirements.
      CJIS: Criminal Justice Information Services (CJIS) Security policies.
      FEDRAMP_HIGH: FedRAMP High data protection controls
      FEDRAMP_MODERATE: FedRAMP Moderate data protection controls
      US_REGIONAL_ACCESS: Assured Workloads For US Regions data protection
        controls
      HIPAA: Health Insurance Portability and Accountability Act controls
      HITRUST: Health Information Trust Alliance controls
      EU_REGIONS_AND_SUPPORT: Assured Workloads For EU Regions and Support
        controls
      CA_REGIONS_AND_SUPPORT: Assured Workloads For Canada Regions and Support
        controls
      ITAR: International Traffic in Arms Regulations
      AU_REGIONS_AND_US_SUPPORT: Assured Workloads for Australia Regions and
        Support controls
      ASSURED_WORKLOADS_FOR_PARTNERS: Assured Workloads for Partners;
      ISR_REGIONS: Assured Workloads for Israel
      ISR_REGIONS_AND_SUPPORT: Assured Workloads for Israel Regions
      CA_PROTECTED_B: Assured Workloads for Canada Protected B regime
      IL5: Information protection as per DoD IL5 requirements.
      IL2: Information protection as per DoD IL2 requirements.
      JP_REGIONS_AND_SUPPORT: Assured Workloads for Japan Regions
      KSA_REGIONS_AND_SUPPORT_WITH_SOVEREIGNTY_CONTROLS: KSA R5 Controls.
      FREE_REGIONS: Assured Workloads Free Regions
    """
        COMPLIANCE_REGIME_UNSPECIFIED = 0
        IL4 = 1
        CJIS = 2
        FEDRAMP_HIGH = 3
        FEDRAMP_MODERATE = 4
        US_REGIONAL_ACCESS = 5
        HIPAA = 6
        HITRUST = 7
        EU_REGIONS_AND_SUPPORT = 8
        CA_REGIONS_AND_SUPPORT = 9
        ITAR = 10
        AU_REGIONS_AND_US_SUPPORT = 11
        ASSURED_WORKLOADS_FOR_PARTNERS = 12
        ISR_REGIONS = 13
        ISR_REGIONS_AND_SUPPORT = 14
        CA_PROTECTED_B = 15
        IL5 = 16
        IL2 = 17
        JP_REGIONS_AND_SUPPORT = 18
        KSA_REGIONS_AND_SUPPORT_WITH_SOVEREIGNTY_CONTROLS = 19
        FREE_REGIONS = 20

    class KajEnrollmentStateValueValuesEnum(_messages.Enum):
        """Output only. Represents the KAJ enrollment state of the given
    workload.

    Values:
      KAJ_ENROLLMENT_STATE_UNSPECIFIED: Default State for KAJ Enrollment.
      KAJ_ENROLLMENT_STATE_PENDING: Pending State for KAJ Enrollment.
      KAJ_ENROLLMENT_STATE_COMPLETE: Complete State for KAJ Enrollment.
    """
        KAJ_ENROLLMENT_STATE_UNSPECIFIED = 0
        KAJ_ENROLLMENT_STATE_PENDING = 1
        KAJ_ENROLLMENT_STATE_COMPLETE = 2

    class PartnerValueValuesEnum(_messages.Enum):
        """Optional. Partner regime associated with this workload.

    Values:
      PARTNER_UNSPECIFIED: <no description>
      LOCAL_CONTROLS_BY_S3NS: Enum representing S3NS (Thales) partner.
      SOVEREIGN_CONTROLS_BY_T_SYSTEMS: Enum representing T_SYSTEM (TSI)
        partner.
      SOVEREIGN_CONTROLS_BY_SIA_MINSAIT: Enum representing SIA_MINSAIT (Indra)
        partner.
      SOVEREIGN_CONTROLS_BY_PSN: Enum representing PSN (TIM) partner.
    """
        PARTNER_UNSPECIFIED = 0
        LOCAL_CONTROLS_BY_S3NS = 1
        SOVEREIGN_CONTROLS_BY_T_SYSTEMS = 2
        SOVEREIGN_CONTROLS_BY_SIA_MINSAIT = 3
        SOVEREIGN_CONTROLS_BY_PSN = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels applied to the workload.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    billingAccount = _messages.StringField(1)
    complianceRegime = _messages.EnumField('ComplianceRegimeValueValuesEnum', 2)
    complianceStatus = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadComplianceStatus', 3)
    compliantButDisallowedServices = _messages.StringField(4, repeated=True)
    createTime = _messages.StringField(5)
    displayName = _messages.StringField(6)
    ekmProvisioningResponse = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadEkmProvisioningResponse', 7)
    enableSovereignControls = _messages.BooleanField(8)
    etag = _messages.StringField(9)
    kajEnrollmentState = _messages.EnumField('KajEnrollmentStateValueValuesEnum', 10)
    kmsSettings = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadKMSSettings', 11)
    labels = _messages.MessageField('LabelsValue', 12)
    name = _messages.StringField(13)
    partner = _messages.EnumField('PartnerValueValuesEnum', 14)
    partnerPermissions = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadPartnerPermissions', 15)
    provisionedResourcesParent = _messages.StringField(16)
    resourceMonitoringEnabled = _messages.BooleanField(17)
    resourceSettings = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadResourceSettings', 18, repeated=True)
    resources = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadResourceInfo', 19, repeated=True)
    saaEnrollmentResponse = _messages.MessageField('GoogleCloudAssuredworkloadsV1WorkloadSaaEnrollmentResponse', 20)
    violationNotificationsEnabled = _messages.BooleanField(21)