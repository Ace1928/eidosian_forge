import glob
import os
import re
import docutils.core
from osprofiler.tests import test
def _check_titles(self, filename, expect, actual):
    missing_sections = [x for x in expect.keys() if x not in actual.keys()]
    extra_sections = [x for x in actual.keys() if x not in expect.keys()]
    msgs = []
    if len(missing_sections) > 0:
        msgs.append('Missing sections: %s' % missing_sections)
    if len(extra_sections) > 0:
        msgs.append('Extra sections: %s' % extra_sections)
    for section in expect.keys():
        missing_subsections = [x for x in expect[section] if x not in actual.get(section, {})]
        if len(missing_subsections) > 0:
            msgs.append("Section '%s' is missing subsections: %s" % (section, missing_subsections))
    if len(msgs) > 0:
        self.fail("While checking '%s':\n  %s" % (filename, '\n  '.join(msgs)))