class UnexpectedSmartServerResponse(BzrError):
    _fmt = 'Could not understand response from smart server: %(response_tuple)r'

    def __init__(self, response_tuple):
        self.response_tuple = response_tuple