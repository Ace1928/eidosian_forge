from boto.exception import JSONResponseError
class DBSubnetGroupNotFound(JSONResponseError):
    pass