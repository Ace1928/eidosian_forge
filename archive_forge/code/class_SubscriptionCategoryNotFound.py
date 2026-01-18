from boto.exception import JSONResponseError
class SubscriptionCategoryNotFound(JSONResponseError):
    pass