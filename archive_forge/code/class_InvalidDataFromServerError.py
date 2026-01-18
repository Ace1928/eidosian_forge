class InvalidDataFromServerError(InvalidDataError, CommunicationError):
    """Data received from the server is malformed."""