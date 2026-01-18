import gast
import termcolor
def _color(self, string, color, attrs=None):
    if self.color:
        return termcolor.colored(string, color, attrs=attrs)
    return string