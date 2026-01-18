from boto.exception import JSONResponseError
class HsmClientCertificateNotFound(JSONResponseError):
    pass