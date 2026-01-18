class StreamStartEvent(Event):
    __slots__ = ('encoding',)

    def __init__(self, start_mark=None, end_mark=None, encoding=None, comment=None):
        Event.__init__(self, start_mark, end_mark, comment)
        self.encoding = encoding