from string import Template
import sys
def _gen_attr_names(self):
    src = '    attr_names = (' + ''.join(('%r, ' % nm for nm in self.attr)) + ')'
    return src