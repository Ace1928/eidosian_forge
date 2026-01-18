from boto.exception import JSONResponseError
class ReservedDBInstanceAlreadyExists(JSONResponseError):
    pass