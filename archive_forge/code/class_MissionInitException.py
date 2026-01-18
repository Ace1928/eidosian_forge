class MissionInitException(Exception):
    """An exception thrown when a mission fails to initialize

    Args:
        message (str): The exception message.
    """

    def __init__(self, message):
        super(MissionInitException, self).__init__(message)