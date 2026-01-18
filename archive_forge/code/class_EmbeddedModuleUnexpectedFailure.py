class EmbeddedModuleUnexpectedFailure(Exception):

    def __init__(self, msg):
        self._message = msg

    def get_message(self):
        return self._message

    def __repr__(self):
        return repr(self.get_message())

    def __str__(self):
        return str(self.get_message())