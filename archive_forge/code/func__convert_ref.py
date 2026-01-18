import _markupbase
import re
def _convert_ref(self, match):
    if match.group(2):
        return self.convert_charref(match.group(2)) or '&#%s%s' % match.groups()[1:]
    elif match.group(3):
        return self.convert_entityref(match.group(1)) or '&%s;' % match.group(1)
    else:
        return '&%s' % match.group(1)