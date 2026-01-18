from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeSubnetworksPatchRequest(_messages.Message):
    """A ComputeSubnetworksPatchRequest object.

  Fields:
    drainTimeoutSeconds: The drain timeout specifies the upper bound in
      seconds on the amount of time allowed to drain connections from the
      current ACTIVE subnetwork to the current BACKUP subnetwork. The drain
      timeout is only applicable when the following conditions are true: - the
      subnetwork being patched has purpose = INTERNAL_HTTPS_LOAD_BALANCER -
      the subnetwork being patched has role = BACKUP - the patch request is
      setting the role to ACTIVE. Note that after this patch operation the
      roles of the ACTIVE and BACKUP subnetworks will be swapped.
    project: Project ID for this request.
    region: Name of the region scoping this request.
    requestId: An optional request ID to identify requests. Specify a unique
      request ID so that if you must retry your request, the server will know
      to ignore the request if it has already been completed. For example,
      consider a situation where you make an initial request and the request
      times out. If you make the request again with the same request ID, the
      server can check if original operation with the same request ID was
      received, and if so, will ignore the second request. This prevents
      clients from accidentally creating duplicate commitments. The request ID
      must be a valid UUID with the exception that zero UUID is not supported
      ( 00000000-0000-0000-0000-000000000000).
    subnetwork: Name of the Subnetwork resource to patch.
    subnetworkResource: A Subnetwork resource to be passed as the request
      body.
  """
    drainTimeoutSeconds = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    subnetwork = _messages.StringField(5, required=True)
    subnetworkResource = _messages.MessageField('Subnetwork', 6)