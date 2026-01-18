from decimal import Decimal
from boto.compat import filter, map
class GetFeedSubmissionListResult(ResponseElement):
    FeedSubmissionInfo = ElementList(FeedSubmissionInfo)