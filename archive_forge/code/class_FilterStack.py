from sqlparse import lexer
from sqlparse.engine import grouping
from sqlparse.engine.statement_splitter import StatementSplitter
class FilterStack(object):

    def __init__(self):
        self.preprocess = []
        self.stmtprocess = []
        self.postprocess = []
        self._grouping = False

    def enable_grouping(self):
        self._grouping = True

    def run(self, sql, encoding=None):
        stream = lexer.tokenize(sql, encoding)
        for filter_ in self.preprocess:
            stream = filter_.process(stream)
        stream = StatementSplitter().process(stream)
        for stmt in stream:
            if self._grouping:
                stmt = grouping.group(stmt)
            for filter_ in self.stmtprocess:
                filter_.process(stmt)
            for filter_ in self.postprocess:
                stmt = filter_.process(stmt)
            yield stmt