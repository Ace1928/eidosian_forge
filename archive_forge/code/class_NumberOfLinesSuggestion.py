import xml.sax.saxutils
class NumberOfLinesSuggestion(object):
    template = '<NumberOfLinesSuggestion>%(num_lines)s</NumberOfLinesSuggestion>'

    def __init__(self, num_lines=1):
        self.num_lines = num_lines

    def get_as_xml(self):
        num_lines = self.num_lines
        return self.template % vars()