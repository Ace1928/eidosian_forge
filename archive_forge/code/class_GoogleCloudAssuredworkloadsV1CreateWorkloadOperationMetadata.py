from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1CreateWorkloadOperationMetadata(_messages.Message):
    """Operation metadata to give request details of CreateWorkload.

  Enums:
    ComplianceRegimeValueValuesEnum: Optional. Compliance controls that should
      be applied to the resources managed by the workload.

  Fields:
    complianceRegime: Optional. Compliance controls that should be applied to
      the resources managed by the workload.
    createTime: Optional. Time when the operation was created.
    displayName: Optional. The display name of the workload.
    parent: Optional. The parent of the workload.
  """

    class ComplianceRegimeValueValuesEnum(_messages.Enum):
        """Optional. Compliance controls that should be applied to the resources
    managed by the workload.

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
    complianceRegime = _messages.EnumField('ComplianceRegimeValueValuesEnum', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    parent = _messages.StringField(4)