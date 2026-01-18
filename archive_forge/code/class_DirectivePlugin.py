import re
class DirectivePlugin:

    def __init__(self):
        self.parser = None

    def parse_options(self, m: re.Match):
        return self.parser.parse_options(m)

    def parse_type(self, m: re.Match):
        return self.parser.parse_type(m)

    def parse_title(self, m: re.Match):
        return self.parser.parse_title(m)

    def parse_content(self, m: re.Match):
        return self.parser.parse_content(m)

    def parse_tokens(self, block, text, state):
        return self.parser.parse_tokens(block, text, state)

    def parse(self, block, m, state):
        raise NotImplementedError()

    def __call__(self, md):
        raise NotImplementedError()