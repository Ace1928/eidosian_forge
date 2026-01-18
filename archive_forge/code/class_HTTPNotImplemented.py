class HTTPNotImplemented(ClientException):
    """HTTP 501
    - Not Implemented: the server does not support this operation.
    """
    http_status = 501
    message = 'Not Implemented'