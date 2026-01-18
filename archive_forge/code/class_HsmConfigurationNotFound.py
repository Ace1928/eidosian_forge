from boto.exception import JSONResponseError
class HsmConfigurationNotFound(JSONResponseError):
    pass