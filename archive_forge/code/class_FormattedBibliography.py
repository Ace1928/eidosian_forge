from __future__ import unicode_literals
class FormattedBibliography(object):

    def __init__(self, entries, style, preamble=''):
        self.entries = list(entries)
        self.style = style
        self.preamble = preamble

    def __iter__(self):
        return iter(self.entries)

    def get_longest_label(self):
        label_style = self.style.label_style
        return label_style.get_longest_label(self.entries)