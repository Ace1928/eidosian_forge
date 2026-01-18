from decimal import Decimal
from boto.compat import filter, map
class SubmitFeedResult(ResponseElement):
    FeedSubmissionInfo = Element(FeedSubmissionInfo)