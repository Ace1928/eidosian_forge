from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Organization(_messages.Message):
    """A GoogleCloudApigeeV1Organization object.

  Enums:
    BillingTypeValueValuesEnum: Billing type of the Apigee organization. See
      [Apigee pricing](https://cloud.google.com/apigee/pricing).
    ReleaseChannelValueValuesEnum: Release channel influences the timing and
      frequency of new updates to the Apigee runtimes instances of the
      organization. It can be either STABLE, REGULAR, or RAPID. It can be
      selected during creation of the Organization and it can also be updated
      later on. Each channel has its own combination of release frequency and
      stability expectations. The RAPID channel will get updates early and
      more often. The REGULAR channel will get updates after being validated
      in the RAPID channel for some time. The STABLE channel will get updates
      after being validated in the REGULAR channel for some time.
    RuntimeTypeValueValuesEnum: Required. Runtime type of the Apigee
      organization based on the Apigee subscription purchased.
    StateValueValuesEnum: Output only. State of the organization. Values other
      than ACTIVE means the resource is not ready to use.
    SubscriptionPlanValueValuesEnum: Output only. Subscription plan that the
      customer has purchased. Output only.
    SubscriptionTypeValueValuesEnum: Output only. DEPRECATED: This will
      eventually be replaced by BillingType. Subscription type of the Apigee
      organization. Valid values include trial (free, limited, and for
      evaluation purposes only) or paid (full subscription has been
      purchased). See [Apigee
      pricing](https://cloud.google.com/apigee/pricing/).
    TypeValueValuesEnum: Not used by Apigee.

  Fields:
    addonsConfig: Addon configurations of the Apigee organization.
    analyticsRegion: Required. DEPRECATED: This field will eventually be
      deprecated and replaced with a differently-named field. Primary Google
      Cloud region for analytics data storage. For valid values, see [Create
      an Apigee organization](https://cloud.google.com/apigee/docs/api-
      platform/get-started/create-org).
    apiConsumerDataEncryptionKeyName: Cloud KMS key name used for encrypting
      API consumer data. Required for US/EU regions when
      [BillingType](#BillingType) is `SUBSCRIPTION`. When
      [BillingType](#BillingType) is `EVALUATION` or the region is not US/EU,
      a Google-Managed encryption key will be used. Format:
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`
    apiConsumerDataLocation: This field is needed only for customers with
      control plane in US or EU. Apigee stores some control plane data only in
      single region. This field determines which single region Apigee should
      use. For example: "us-west1" when control plane is in US or "europe-
      west2" when control plane is in EU.
    apigeeProjectId: Output only. Apigee Project ID associated with the
      organization. Use this project to allowlist Apigee in the Service
      Attachment when using private service connect with Apigee.
    attributes: Not used by Apigee.
    authorizedNetwork: Compute Engine network used for Service Networking to
      be peered with Apigee runtime instances. See [Getting started with the
      Service Networking API](https://cloud.google.com/service-
      infrastructure/docs/service-networking/getting-started). Valid only when
      [RuntimeType](#RuntimeType) is set to `CLOUD`. The value must be set
      before the creation of a runtime instance and can be updated only when
      there are no runtime instances. For example: `default`. Apigee also
      supports shared VPC (that is, the host network project is not the same
      as the one that is peering with Apigee). See [Shared VPC
      overview](https://cloud.google.com/vpc/docs/shared-vpc). To use a shared
      VPC network, use the following format: `projects/{host-project-
      id}/{region}/networks/{network-name}`. For example: `projects/my-
      sharedvpc-host/global/networks/mynetwork` **Note:** Not supported for
      Apigee hybrid.
    billingType: Billing type of the Apigee organization. See [Apigee
      pricing](https://cloud.google.com/apigee/pricing).
    caCertificate: Output only. Base64-encoded public certificate for the root
      CA of the Apigee organization. Valid only when
      [RuntimeType](#RuntimeType) is `CLOUD`.
    controlPlaneEncryptionKeyName: Cloud KMS key name used for encrypting
      control plane data that is stored in a multi region. Required when
      [BillingType](#BillingType) is `SUBSCRIPTION`. When
      [BillingType](#BillingType) is `EVALUATION`, a Google-Managed encryption
      key will be used. Format:
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`
    createdAt: Output only. Time that the Apigee organization was created in
      milliseconds since epoch.
    customerName: Not used by Apigee.
    description: Description of the Apigee organization.
    disableVpcPeering: Optional. Flag that specifies whether the VPC Peering
      through Private Google Access should be disabled between the consumer
      network and Apigee. Valid only when RuntimeType is set to CLOUD.
      Required if an authorizedNetwork on the consumer project is not
      provided, in which case the flag should be set to true. The value must
      be set before the creation of any Apigee runtime instance and can be
      updated only when there are no runtime instances. **Note:** Apigee will
      be deprecating the vpc peering model that requires you to provide
      'authorizedNetwork', by making the non-peering model as the default way
      of provisioning Apigee organization in future. So, this will be a
      temporary flag to enable the transition. Not supported for Apigee
      hybrid.
    displayName: Display name for the Apigee organization. Unused, but
      reserved for future use.
    environments: Output only. List of environments in the Apigee
      organization.
    expiresAt: Output only. Time that the Apigee organization is scheduled for
      deletion.
    lastModifiedAt: Output only. Time that the Apigee organization was last
      modified in milliseconds since epoch.
    name: Output only. Name of the Apigee organization.
    portalDisabled: Configuration for the Portals settings.
    projectId: Output only. Project ID associated with the Apigee
      organization.
    properties: Properties defined in the Apigee organization profile.
    releaseChannel: Release channel influences the timing and frequency of new
      updates to the Apigee runtimes instances of the organization. It can be
      either STABLE, REGULAR, or RAPID. It can be selected during creation of
      the Organization and it can also be updated later on. Each channel has
      its own combination of release frequency and stability expectations. The
      RAPID channel will get updates early and more often. The REGULAR channel
      will get updates after being validated in the RAPID channel for some
      time. The STABLE channel will get updates after being validated in the
      REGULAR channel for some time.
    runtimeDatabaseEncryptionKeyName: Cloud KMS key name used for encrypting
      the data that is stored and replicated across runtime instances. Update
      is not allowed after the organization is created. Required when
      [RuntimeType](#RuntimeType) is `CLOUD`. If not specified when
      [RuntimeType](#RuntimeType) is `TRIAL`, a Google-Managed encryption key
      will be used. For example:
      "projects/foo/locations/us/keyRings/bar/cryptoKeys/baz". **Note:** Not
      supported for Apigee hybrid.
    runtimeType: Required. Runtime type of the Apigee organization based on
      the Apigee subscription purchased.
    state: Output only. State of the organization. Values other than ACTIVE
      means the resource is not ready to use.
    subscriptionPlan: Output only. Subscription plan that the customer has
      purchased. Output only.
    subscriptionType: Output only. DEPRECATED: This will eventually be
      replaced by BillingType. Subscription type of the Apigee organization.
      Valid values include trial (free, limited, and for evaluation purposes
      only) or paid (full subscription has been purchased). See [Apigee
      pricing](https://cloud.google.com/apigee/pricing/).
    type: Not used by Apigee.
  """

    class BillingTypeValueValuesEnum(_messages.Enum):
        """Billing type of the Apigee organization. See [Apigee
    pricing](https://cloud.google.com/apigee/pricing).

    Values:
      BILLING_TYPE_UNSPECIFIED: Billing type not specified.
      SUBSCRIPTION: A pre-paid subscription to Apigee.
      EVALUATION: Free and limited access to Apigee for evaluation purposes
        only.
      PAYG: Access to Apigee using a Pay-As-You-Go plan.
    """
        BILLING_TYPE_UNSPECIFIED = 0
        SUBSCRIPTION = 1
        EVALUATION = 2
        PAYG = 3

    class ReleaseChannelValueValuesEnum(_messages.Enum):
        """Release channel influences the timing and frequency of new updates to
    the Apigee runtimes instances of the organization. It can be either
    STABLE, REGULAR, or RAPID. It can be selected during creation of the
    Organization and it can also be updated later on. Each channel has its own
    combination of release frequency and stability expectations. The RAPID
    channel will get updates early and more often. The REGULAR channel will
    get updates after being validated in the RAPID channel for some time. The
    STABLE channel will get updates after being validated in the REGULAR
    channel for some time.

    Values:
      RELEASE_CHANNEL_UNSPECIFIED: Release channel not specified.
      STABLE: Stable release channel.
      REGULAR: Regular release channel.
      RAPID: Rapid release channel.
    """
        RELEASE_CHANNEL_UNSPECIFIED = 0
        STABLE = 1
        REGULAR = 2
        RAPID = 3

    class RuntimeTypeValueValuesEnum(_messages.Enum):
        """Required. Runtime type of the Apigee organization based on the Apigee
    subscription purchased.

    Values:
      RUNTIME_TYPE_UNSPECIFIED: Runtime type not specified.
      CLOUD: Google-managed Apigee runtime.
      HYBRID: User-managed Apigee hybrid runtime.
    """
        RUNTIME_TYPE_UNSPECIFIED = 0
        CLOUD = 1
        HYBRID = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the organization. Values other than ACTIVE means
    the resource is not ready to use.

    Values:
      STATE_UNSPECIFIED: Resource is in an unspecified state.
      CREATING: Resource is being created.
      ACTIVE: Resource is provisioned and ready to use.
      DELETING: The resource is being deleted.
      UPDATING: The resource is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        UPDATING = 4

    class SubscriptionPlanValueValuesEnum(_messages.Enum):
        """Output only. Subscription plan that the customer has purchased. Output
    only.

    Values:
      SUBSCRIPTION_PLAN_UNSPECIFIED: Subscription plan not specified.
      SUBSCRIPTION_2021: Traditional subscription plan.
      SUBSCRIPTION_2024: New subscription plan that provides standard proxy
        and scaled proxy implementation.
    """
        SUBSCRIPTION_PLAN_UNSPECIFIED = 0
        SUBSCRIPTION_2021 = 1
        SUBSCRIPTION_2024 = 2

    class SubscriptionTypeValueValuesEnum(_messages.Enum):
        """Output only. DEPRECATED: This will eventually be replaced by
    BillingType. Subscription type of the Apigee organization. Valid values
    include trial (free, limited, and for evaluation purposes only) or paid
    (full subscription has been purchased). See [Apigee
    pricing](https://cloud.google.com/apigee/pricing/).

    Values:
      SUBSCRIPTION_TYPE_UNSPECIFIED: Subscription type not specified.
      PAID: Full subscription to Apigee has been purchased.
      TRIAL: Subscription to Apigee is free, limited, and used for evaluation
        purposes only.
    """
        SUBSCRIPTION_TYPE_UNSPECIFIED = 0
        PAID = 1
        TRIAL = 2

    class TypeValueValuesEnum(_messages.Enum):
        """Not used by Apigee.

    Values:
      TYPE_UNSPECIFIED: Subscription type not specified.
      TYPE_TRIAL: Subscription to Apigee is free, limited, and used for
        evaluation purposes only.
      TYPE_PAID: Full subscription to Apigee has been purchased. See [Apigee
        pricing](https://cloud.google.com/apigee/pricing/).
      TYPE_INTERNAL: For internal users only.
    """
        TYPE_UNSPECIFIED = 0
        TYPE_TRIAL = 1
        TYPE_PAID = 2
        TYPE_INTERNAL = 3
    addonsConfig = _messages.MessageField('GoogleCloudApigeeV1AddonsConfig', 1)
    analyticsRegion = _messages.StringField(2)
    apiConsumerDataEncryptionKeyName = _messages.StringField(3)
    apiConsumerDataLocation = _messages.StringField(4)
    apigeeProjectId = _messages.StringField(5)
    attributes = _messages.StringField(6, repeated=True)
    authorizedNetwork = _messages.StringField(7)
    billingType = _messages.EnumField('BillingTypeValueValuesEnum', 8)
    caCertificate = _messages.BytesField(9)
    controlPlaneEncryptionKeyName = _messages.StringField(10)
    createdAt = _messages.IntegerField(11)
    customerName = _messages.StringField(12)
    description = _messages.StringField(13)
    disableVpcPeering = _messages.BooleanField(14)
    displayName = _messages.StringField(15)
    environments = _messages.StringField(16, repeated=True)
    expiresAt = _messages.IntegerField(17)
    lastModifiedAt = _messages.IntegerField(18)
    name = _messages.StringField(19)
    portalDisabled = _messages.BooleanField(20)
    projectId = _messages.StringField(21)
    properties = _messages.MessageField('GoogleCloudApigeeV1Properties', 22)
    releaseChannel = _messages.EnumField('ReleaseChannelValueValuesEnum', 23)
    runtimeDatabaseEncryptionKeyName = _messages.StringField(24)
    runtimeType = _messages.EnumField('RuntimeTypeValueValuesEnum', 25)
    state = _messages.EnumField('StateValueValuesEnum', 26)
    subscriptionPlan = _messages.EnumField('SubscriptionPlanValueValuesEnum', 27)
    subscriptionType = _messages.EnumField('SubscriptionTypeValueValuesEnum', 28)
    type = _messages.EnumField('TypeValueValuesEnum', 29)