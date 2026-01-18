class ContainerWebSocketException(Exception):
    """base for all ContainerWebSocket interactive generated exceptions"""

    def __init__(self, wrapped=None, message=None):
        self.wrapped = wrapped
        if message:
            self.message = message
        if wrapped:
            formatted_string = '%s:%s' % (self.message, str(self.wrapped))
        else:
            formatted_string = '%s' % self.message
        super(ContainerWebSocketException, self).__init__(formatted_string)