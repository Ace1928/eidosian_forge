import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _get_titles(self, spec):
    titles = {}
    for node in spec:
        if node.tagname == 'section':
            section = self._get_title(node)
            titles[section['name']] = section['subtitles']
    return titles