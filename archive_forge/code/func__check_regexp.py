import re
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import regexp_span_tokenize
def _check_regexp(self):
    if self._regexp is None:
        self._regexp = re.compile(self._pattern, self._flags)