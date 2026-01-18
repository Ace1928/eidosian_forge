class ErrorFromSmartServer(BzrError):
    """An error was received from a smart server.

    :seealso: UnknownErrorFromSmartServer
    """
    _fmt = 'Error received from smart server: %(error_tuple)r'
    internal_error = True

    def __init__(self, error_tuple):
        self.error_tuple = error_tuple
        try:
            self.error_verb = error_tuple[0]
        except IndexError:
            self.error_verb = None
        self.error_args = error_tuple[1:]