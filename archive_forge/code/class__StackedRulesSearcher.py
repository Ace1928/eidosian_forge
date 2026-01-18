import configobj
from . import bedding, cmdline, errors, globbing, osutils
class _StackedRulesSearcher(_RulesSearcher):

    def __init__(self, searchers):
        """Construct a _RulesSearcher based on a stack of other ones.

        :param searchers: a sequence of searchers.
        """
        self.searchers = searchers

    def get_items(self, path):
        """See _RulesSearcher.get_items."""
        for searcher in self.searchers:
            result = searcher.get_items(path)
            if result:
                return result
        return ()

    def get_selected_items(self, path, names):
        """See _RulesSearcher.get_selected_items."""
        for searcher in self.searchers:
            result = searcher.get_selected_items(path, names)
            if result:
                return result
        return ()