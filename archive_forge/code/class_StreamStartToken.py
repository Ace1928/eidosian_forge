from __future__ import unicode_literals
class StreamStartToken(Token):
    __slots__ = ('encoding',)
    id = '<stream start>'

    def __init__(self, start_mark=None, end_mark=None, encoding=None):
        Token.__init__(self, start_mark, end_mark)
        self.encoding = encoding