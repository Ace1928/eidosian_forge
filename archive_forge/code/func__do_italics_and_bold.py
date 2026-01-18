import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _do_italics_and_bold(self, text):
    if self.extras.get('middle-word-em', True) is False:
        code_friendly_em_re = '(?<=\\b)%s(?=\\b)' % self._code_friendly_em_re
        em_re = '(?<=\\b)%s(?=\\b)' % self._em_re
    else:
        code_friendly_em_re = self._code_friendly_em_re
        em_re = self._em_re
    if 'code-friendly' in self.extras:
        text = self._code_friendly_strong_re.sub('<strong>\\1</strong>', text)
        text = re.sub(code_friendly_em_re, '<em>\\1</em>', text, flags=re.S)
    else:
        text = self._strong_re.sub('<strong>\\2</strong>', text)
        text = re.sub(em_re, '<em>\\2</em>', text, flags=re.S)
    return text