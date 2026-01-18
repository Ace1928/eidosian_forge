from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessApprovalSettings(_messages.Message):
    """Settings on a Project/Folder/Organization related to Access Approval.

  Fields:
    activeKeyVersion: The asymmetric crypto key version to use for signing
      approval requests. Empty active_key_version indicates that a Google-
      managed key should be used for signing. This property will be ignored if
      set by an ancestor of this resource, and new non-empty values may not be
      set.
    ancestorHasActiveKeyVersion: Output only. This field is read only (not
      settable via UpdateAccessApprovalSettings method). If the field is true,
      that indicates that an ancestor of this Project or Folder has set
      active_key_version (this field will always be unset for the organization
      since organizations do not have ancestors).
    enrolledAncestor: Output only. This field is read only (not settable via
      UpdateAccessApprovalSettings method). If the field is true, that
      indicates that at least one service is enrolled for Access Approval in
      one or more ancestors of the Project or Folder (this field will always
      be unset for the organization since organizations do not have
      ancestors).
    enrolledServices: A list of Google Cloud Services for which the given
      resource has Access Approval enrolled. Access requests for the resource
      given by name against any of these services contained here will be
      required to have explicit approval. If name refers to an organization,
      enrollment can be done for individual services. If name refers to a
      folder or project, enrollment can only be done on an all or nothing
      basis. If a cloud_product is repeated in this list, the first entry will
      be honored and all following entries will be discarded. A maximum of 10
      enrolled services will be enforced, to be expanded as the set of
      supported services is expanded.
    invalidKeyVersion: Output only. This field is read only (not settable via
      UpdateAccessApprovalSettings method). If the field is true, that
      indicates that there is some configuration issue with the
      active_key_version configured at this level in the resource hierarchy
      (e.g. it doesn't exist or the Access Approval service account doesn't
      have the correct permissions on it, etc.) This key version is not
      necessarily the effective key version at this level, as key versions are
      inherited top-down.
    name: The resource name of the settings. Format is one of: *
      "projects/{project}/accessApprovalSettings" *
      "folders/{folder}/accessApprovalSettings" *
      "organizations/{organization}/accessApprovalSettings"
    notificationEmails: A list of email addresses to which notifications
      relating to approval requests should be sent. Notifications relating to
      a resource will be sent to all emails in the settings of ancestor
      resources of that resource. A maximum of 50 email addresses are allowed.
    notificationPubsubTopic: Optional. A pubsub topic to which notifications
      relating to approval requests should be sent.
    preferNoBroadApprovalRequests: This preference is communicated to Google
      personnel when sending an approval request but can be overridden if
      necessary.
    preferredRequestExpirationDays: This preference is shared with Google
      personnel, but can be overridden if said personnel deems necessary. The
      approver ultimately can set the expiration at approval time.
  """
    activeKeyVersion = _messages.StringField(1)
    ancestorHasActiveKeyVersion = _messages.BooleanField(2)
    enrolledAncestor = _messages.BooleanField(3)
    enrolledServices = _messages.MessageField('EnrolledService', 4, repeated=True)
    invalidKeyVersion = _messages.BooleanField(5)
    name = _messages.StringField(6)
    notificationEmails = _messages.StringField(7, repeated=True)
    notificationPubsubTopic = _messages.StringField(8)
    preferNoBroadApprovalRequests = _messages.BooleanField(9)
    preferredRequestExpirationDays = _messages.IntegerField(10, variant=_messages.Variant.INT32)