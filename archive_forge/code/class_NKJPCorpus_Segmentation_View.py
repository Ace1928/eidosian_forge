import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
class NKJPCorpus_Segmentation_View(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with
    ann_segmentation.xml files in NKJP corpus.
    """

    def __init__(self, filename, **kwargs):
        self.tagspec = '.*p/.*s'
        self.text_view = NKJPCorpus_Text_View(filename, mode=NKJPCorpus_Text_View.SENTS_MODE)
        self.text_view.handle_query()
        self.xml_tool = XML_Tool(filename, 'ann_segmentation.xml')
        XMLCorpusView.__init__(self, self.xml_tool.build_preprocessed_file(), self.tagspec)

    def get_segm_id(self, example_word):
        return example_word.split('(')[1].split(',')[0]

    def get_sent_beg(self, beg_word):
        return int(beg_word.split(',')[1])

    def get_sent_end(self, end_word):
        splitted = end_word.split(')')[0].split(',')
        return int(splitted[1]) + int(splitted[2])

    def get_sentences(self, sent_segm):
        id = self.get_segm_id(sent_segm[0])
        segm = self.text_view.segm_dict[id]
        beg = self.get_sent_beg(sent_segm[0])
        end = self.get_sent_end(sent_segm[len(sent_segm) - 1])
        return segm[beg:end]

    def remove_choice(self, segm):
        ret = []
        prev_txt_end = -1
        prev_txt_nr = -1
        for word in segm:
            txt_nr = self.get_segm_id(word)
            if self.get_sent_beg(word) > prev_txt_end - 1 or prev_txt_nr != txt_nr:
                ret.append(word)
                prev_txt_end = self.get_sent_end(word)
            prev_txt_nr = txt_nr
        return ret

    def handle_query(self):
        try:
            self._open()
            sentences = []
            while True:
                sent_segm = XMLCorpusView.read_block(self, self._stream)
                if len(sent_segm) == 0:
                    break
                for segm in sent_segm:
                    segm = self.remove_choice(segm)
                    sentences.append(self.get_sentences(segm))
            self.close()
            self.xml_tool.remove_preprocessed_file()
            return sentences
        except Exception as e:
            self.xml_tool.remove_preprocessed_file()
            raise Exception from e

    def handle_elt(self, elt, context):
        ret = []
        for seg in elt:
            ret.append(seg.get('corresp'))
        return ret