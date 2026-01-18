class YaqlGrammarException(YaqlParsingException):

    def __init__(self, expr, value, position):
        if position is None:
            msg = u'Parse error: unexpected end of statement'
        else:
            msg = u"Parse error: unexpected '{0}' at position {1} of expression '{2}'".format(value, position, expr)
        super(YaqlGrammarException, self).__init__(value, position, msg)