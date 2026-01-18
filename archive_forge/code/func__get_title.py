import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _get_title(self, section_tree):
    section = {'subtitles': []}
    for node in section_tree:
        if node.tagname == 'title':
            section['name'] = node.rawsource
        elif node.tagname == 'section':
            subsection = self._get_title(node)
            section['subtitles'].append(subsection['name'])
    return section