from decimal import Decimal
from boto.compat import filter, map
class GetFulfillmentPreviewResult(ResponseElement):
    FulfillmentPreviews = MemberList(FulfillmentPreview)