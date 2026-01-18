class OverLimit(ClientException):
    """HTTP 413
    - Over limit: you're over the API limits for this time period.
    """
    http_status = 413
    message = 'Over limit'