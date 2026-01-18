import re
class LabelSelectorFilter(object):

    def __init__(self, label_selectors):
        self.selectors = [Selector(data) for data in label_selectors]

    def isMatching(self, definition):
        if 'metadata' not in definition or 'labels' not in definition['metadata']:
            return False
        labels = definition['metadata']['labels']
        if not isinstance(labels, dict):
            return None
        return all((sel.isMatch(labels) for sel in self.selectors))