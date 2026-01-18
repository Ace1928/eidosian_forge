import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _html_class_str_from_tag(self, tag):
    """Get the appropriate ' class="..."' string (note the leading
        space), if any, for the given tag.
        """
    if 'html-classes' not in self.extras:
        return ''
    try:
        html_classes_from_tag = self.extras['html-classes']
    except TypeError:
        return ''
    else:
        if isinstance(html_classes_from_tag, dict):
            if tag in html_classes_from_tag:
                return ' class="%s"' % html_classes_from_tag[tag]
    return ''