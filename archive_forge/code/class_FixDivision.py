from lib2to3 import fixer_base
from libfuturize.fixer_util import token, future_import
class FixDivision(fixer_base.BaseFix):
    run_order = 4

    def match(self, node):
        u"""
        Since the tree needs to be fixed once and only once if and only if it
        matches, then we can start discarding matches after we make the first.
        """
        return match_division(node)

    def transform(self, node, results):
        future_import(u'division', node)