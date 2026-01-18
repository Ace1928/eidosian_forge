from boto.exception import JSONResponseError
class CopyToRegionDisabled(JSONResponseError):
    pass