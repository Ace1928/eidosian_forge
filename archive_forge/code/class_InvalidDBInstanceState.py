from boto.exception import JSONResponseError
class InvalidDBInstanceState(JSONResponseError):
    pass