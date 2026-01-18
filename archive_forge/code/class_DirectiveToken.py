from __future__ import unicode_literals
class DirectiveToken(Token):
    __slots__ = ('name', 'value')
    id = '<directive>'

    def __init__(self, name, value, start_mark, end_mark):
        Token.__init__(self, start_mark, end_mark)
        self.name = name
        self.value = value