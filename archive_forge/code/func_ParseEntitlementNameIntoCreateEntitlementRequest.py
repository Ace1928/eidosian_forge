from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ParseEntitlementNameIntoCreateEntitlementRequest(unused_ref, args, req):
    """Modify request hook to parse the entitlement name into a CreateEntitlementRequest."""
    entitlement = args.CONCEPTS.entitlement.Parse()
    req.parent = entitlement.result.Parent().RelativeName()
    req.entitlementId = entitlement.result.Name()
    return req