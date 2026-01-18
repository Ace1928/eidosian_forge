import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _check_trailing_spaces(self, tpl, raw):
    for i, line in enumerate(raw.split('\n')):
        trailing_spaces = re.findall(' +$', line)
        self.assertEqual(len(trailing_spaces), 0, 'Found trailing spaces on line %s of %s' % (i + 1, tpl))