import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
class NKJPCorpus_Text_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    text.xml files in NKJP corpus.
    """
    SENTS_MODE = 0
    RAW_MODE = 1

    def __init__(self, filename, **kwargs):
        self.mode = kwargs.pop('mode', 0)
        self.tagspec = '.*/div/ab'
        self.segm_dict = dict()
        self.xml_tool = XML_Tool(filename, 'text.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def handle_query(self):
        try:
            self._open()
            x = self.read_block(self._stream)
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return x
        except Exception as e:
            self.xml_tool.remove_preprocessed_file()
            raise Exception from e

    def read_block(self, stream, tagspec=None, elt_handler=None):
        """
        Returns text as a list of sentences.
        """
        txt = []
        while True:
            segm = XMLCorpusView.read_block(self, stream)
            if len(segm) == 0:
                break
            for part in segm:
                txt.append(part)
        return [' '.join([segm for segm in txt])]

    def get_segm_id(self, elt):
        for attr in elt.attrib:
            if attr.endswith('id'):
                return elt.get(attr)

    def handle_elt(self, elt, context):
        if self.mode is NKJPCorpus_Text_View.SENTS_MODE:
            self.segm_dict[self.get_segm_id(elt)] = elt.text
        return elt.text