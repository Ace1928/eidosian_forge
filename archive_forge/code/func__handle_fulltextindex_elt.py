import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def _handle_fulltextindex_elt(self, elt, tagspec=None):
    """
        Extracts corpus/document info from the fulltextIndex.xml file.

        Note that this function "flattens" the information contained
        in each of the "corpus" elements, so that each "document"
        element will contain attributes for the corpus and
        corpusid. Also, each of the "document" items will contain a
        new attribute called "filename" that is the base file name of
        the xml file for the document in the "fulltext" subdir of the
        Framenet corpus.
        """
    ftinfo = self._load_xml_attributes(AttrDict(), elt)
    corpname = ftinfo.name
    corpid = ftinfo.ID
    retlist = []
    for sub in elt:
        if sub.tag.endswith('document'):
            doc = self._load_xml_attributes(AttrDict(), sub)
            if 'name' in doc:
                docname = doc.name
            else:
                docname = doc.description
            doc.filename = f'{corpname}__{docname}.xml'
            doc.URL = self._fnweb_url + '/' + self._fulltext_dir + '/' + doc.filename
            doc.corpname = corpname
            doc.corpid = corpid
            retlist.append(doc)
    return retlist