class EnvException(Exception):
    """A special exception thrown in the creation of an environment's Malmo mission XML.

    Args:
        message (str): The exception message.
    """

    def __init__(self, message):
        super(EnvException, self).__init__(message)