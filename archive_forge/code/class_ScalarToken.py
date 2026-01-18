from __future__ import unicode_literals
class ScalarToken(Token):
    __slots__ = ('value', 'plain', 'style')
    id = '<scalar>'

    def __init__(self, value, plain, start_mark, end_mark, style=None):
        Token.__init__(self, start_mark, end_mark)
        self.value = value
        self.plain = plain
        self.style = style