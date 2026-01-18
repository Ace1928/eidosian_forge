class BatchResults(object):
    """
    A container for the results of a send_message_batch request.

    :ivar results: A list of successful results.  Each item in the
        list will be an instance of :class:`ResultEntry`.

    :ivar errors: A list of unsuccessful results.  Each item in the
        list will be an instance of :class:`ResultEntry`.
    """

    def __init__(self, parent):
        self.parent = parent
        self.results = []
        self.errors = []

    def startElement(self, name, attrs, connection):
        if name.endswith('MessageBatchResultEntry'):
            entry = ResultEntry()
            self.results.append(entry)
            return entry
        if name == 'BatchResultErrorEntry':
            entry = ResultEntry()
            self.errors.append(entry)
            return entry
        return None

    def endElement(self, name, value, connection):
        setattr(self, name, value)