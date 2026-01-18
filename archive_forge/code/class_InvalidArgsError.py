from googlecloudsdk.core import exceptions
class InvalidArgsError(exceptions.Error):

    def __init__(self, error_message):
        message = '{}'.format(error_message)
        super(InvalidArgsError, self).__init__(message)