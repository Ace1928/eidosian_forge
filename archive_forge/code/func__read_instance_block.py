import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
def _read_instance_block(self, stream, instance_filter=lambda inst: True):
    block = []
    for i in range(100):
        line = stream.readline().strip()
        if line:
            inst = PropbankInstance.parse(line, self._parse_fileid_xform, self._parse_corpus)
            if instance_filter(inst):
                block.append(inst)
    return block