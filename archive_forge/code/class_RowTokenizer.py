import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class RowTokenizer(object):

    def __init__(self):
        self._table = UnknownTable()
        self._splitter = RowSplitter()
        testcases = TestCaseTable()
        settings = SettingTable(testcases.set_default_template)
        variables = VariableTable()
        keywords = KeywordTable()
        self._tables = {'settings': settings, 'setting': settings, 'metadata': settings, 'variables': variables, 'variable': variables, 'testcases': testcases, 'testcase': testcases, 'keywords': keywords, 'keyword': keywords, 'userkeywords': keywords, 'userkeyword': keywords}

    def tokenize(self, row):
        commented = False
        heading = False
        for index, value in enumerate(self._splitter.split(row)):
            index, separator = divmod(index - 1, 2)
            if value.startswith('#'):
                commented = True
            elif index == 0 and value.startswith('*'):
                self._table = self._start_table(value)
                heading = True
            for value, token in self._tokenize(value, index, commented, separator, heading):
                yield (value, token)
        self._table.end_row()

    def _start_table(self, header):
        name = normalize(header, remove='*')
        return self._tables.get(name, UnknownTable())

    def _tokenize(self, value, index, commented, separator, heading):
        if commented:
            yield (value, COMMENT)
        elif separator:
            yield (value, SEPARATOR)
        elif heading:
            yield (value, HEADING)
        else:
            for value, token in self._table.tokenize(value, index):
                yield (value, token)