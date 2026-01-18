import re
import sgmllib
class EndBracketMatch:

    def __init__(self, match):
        self.match = match

    def start(self, n):
        return self.match.end(n)