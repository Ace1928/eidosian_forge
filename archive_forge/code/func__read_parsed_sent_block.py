import os
import re
from collections import defaultdict
from itertools import chain
from nltk.corpus.reader.util import *
from nltk.data import FileSystemPathPointer, PathPointer, ZipFilePathPointer
def _read_parsed_sent_block(self, stream):
    return list(filter(None, [self._parse(t) for t in self._read_block(stream)]))