from pyparsing import Word, alphanums, Keyword, Group, Combine, Forward, Suppress, OneOrMore, oneOf
class SearchQueryParser:

    def __init__(self):
        self._methods = {'and': self.evaluateAnd, 'or': self.evaluateOr, 'not': self.evaluateNot, 'parenthesis': self.evaluateParenthesis, 'quotes': self.evaluateQuotes, 'word': self.evaluateWord, 'wordwildcard': self.evaluateWordWildcard}
        self._parser = self.parser()

    def parser(self):
        """
        This function returns a parser.
        The grammar should be like most full text search engines (Google, Tsearch, Lucene).

        Grammar:
        - a query consists of alphanumeric words, with an optional '*' wildcard
          at the end of a word
        - a sequence of words between quotes is a literal string
        - words can be used together by using operators ('and' or 'or')
        - words with operators can be grouped with parenthesis
        - a word or group of words can be preceded by a 'not' operator
        - the 'and' operator precedes an 'or' operator
        - if an operator is missing, use an 'and' operator
        """
        operatorOr = Forward()
        operatorWord = Group(Combine(Word(alphanums) + Suppress('*'))).setResultsName('wordwildcard') | Group(Word(alphanums)).setResultsName('word')
        operatorQuotesContent = Forward()
        operatorQuotesContent << (operatorWord + operatorQuotesContent | operatorWord)
        operatorQuotes = Group(Suppress('"') + operatorQuotesContent + Suppress('"')).setResultsName('quotes') | operatorWord
        operatorParenthesis = Group(Suppress('(') + operatorOr + Suppress(')')).setResultsName('parenthesis') | operatorQuotes
        operatorNot = Forward()
        operatorNot << (Group(Suppress(Keyword('not', caseless=True)) + operatorNot).setResultsName('not') | operatorParenthesis)
        operatorAnd = Forward()
        operatorAnd << (Group(operatorNot + Suppress(Keyword('and', caseless=True)) + operatorAnd).setResultsName('and') | Group(operatorNot + OneOrMore(~oneOf('and or') + operatorAnd)).setResultsName('and') | operatorNot)
        operatorOr << (Group(operatorAnd + Suppress(Keyword('or', caseless=True)) + operatorOr).setResultsName('or') | operatorAnd)
        return operatorOr.parseString

    def evaluateAnd(self, argument):
        return self.evaluate(argument[0]).intersection(self.evaluate(argument[1]))

    def evaluateOr(self, argument):
        return self.evaluate(argument[0]).union(self.evaluate(argument[1]))

    def evaluateNot(self, argument):
        return self.GetNot(self.evaluate(argument[0]))

    def evaluateParenthesis(self, argument):
        return self.evaluate(argument[0])

    def evaluateQuotes(self, argument):
        """Evaluate quoted strings

        First is does an 'and' on the indidual search terms, then it asks the
        function GetQuoted to only return the subset of ID's that contain the
        literal string.
        """
        r = set()
        search_terms = []
        for item in argument:
            search_terms.append(item[0])
            if len(r) == 0:
                r = self.evaluate(item)
            else:
                r = r.intersection(self.evaluate(item))
        return self.GetQuotes(' '.join(search_terms), r)

    def evaluateWord(self, argument):
        return self.GetWord(argument[0])

    def evaluateWordWildcard(self, argument):
        return self.GetWordWildcard(argument[0])

    def evaluate(self, argument):
        return self._methods[argument.getName()](argument)

    def Parse(self, query):
        return self.evaluate(self._parser(query)[0])

    def GetWord(self, word):
        return set()

    def GetWordWildcard(self, word):
        return set()

    def GetQuotes(self, search_string, tmp_result):
        return set()

    def GetNot(self, not_set):
        return set().difference(not_set)