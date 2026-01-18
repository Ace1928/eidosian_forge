import os
import coverage
from kivy.lang.parser import Parser
class KivyFileReporter(coverage.plugin.FileReporter):

    def lines(self):
        with open(self.filename) as fh:
            source = fh.read()
        parser = CoverageKVParser(content=source, filename=self.filename)
        return parser.get_coverage_lines()