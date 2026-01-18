from boto.exception import JSONResponseError
class SubscriptionAlreadyExist(JSONResponseError):
    pass