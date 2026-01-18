import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
class MTEFileReader:
    """
    Class for loading the content of the multext-east corpus. It
    parses the xml files and does some tag-filtering depending on the
    given method parameters.
    """
    ns = {'tei': 'https://www.tei-c.org/ns/1.0', 'xml': 'https://www.w3.org/XML/1998/namespace'}
    tag_ns = '{https://www.tei-c.org/ns/1.0}'
    xml_ns = '{https://www.w3.org/XML/1998/namespace}'
    word_path = 'TEI/text/body/div/div/p/s/(w|c)'
    sent_path = 'TEI/text/body/div/div/p/s'
    para_path = 'TEI/text/body/div/div/p'

    def __init__(self, file_path):
        self.__file_path = file_path

    @classmethod
    def _word_elt(cls, elt, context):
        return elt.text

    @classmethod
    def _sent_elt(cls, elt, context):
        return [cls._word_elt(w, None) for w in xpath(elt, '*', cls.ns)]

    @classmethod
    def _para_elt(cls, elt, context):
        return [cls._sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]

    @classmethod
    def _tagged_word_elt(cls, elt, context):
        if 'ana' not in elt.attrib:
            return (elt.text, '')
        if cls.__tags == '' and cls.__tagset == 'msd':
            return (elt.text, elt.attrib['ana'])
        elif cls.__tags == '' and cls.__tagset == 'universal':
            return (elt.text, MTETagConverter.msd_to_universal(elt.attrib['ana']))
        else:
            tags = re.compile('^' + re.sub('-', '.', cls.__tags) + '.*$')
            if tags.match(elt.attrib['ana']):
                if cls.__tagset == 'msd':
                    return (elt.text, elt.attrib['ana'])
                else:
                    return (elt.text, MTETagConverter.msd_to_universal(elt.attrib['ana']))
            else:
                return None

    @classmethod
    def _tagged_sent_elt(cls, elt, context):
        return list(filter(lambda x: x is not None, [cls._tagged_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]))

    @classmethod
    def _tagged_para_elt(cls, elt, context):
        return list(filter(lambda x: x is not None, [cls._tagged_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]))

    @classmethod
    def _lemma_word_elt(cls, elt, context):
        if 'lemma' not in elt.attrib:
            return (elt.text, '')
        else:
            return (elt.text, elt.attrib['lemma'])

    @classmethod
    def _lemma_sent_elt(cls, elt, context):
        return [cls._lemma_word_elt(w, None) for w in xpath(elt, '*', cls.ns)]

    @classmethod
    def _lemma_para_elt(cls, elt, context):
        return [cls._lemma_sent_elt(s, None) for s in xpath(elt, '*', cls.ns)]

    def words(self):
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._word_elt)

    def sents(self):
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._sent_elt)

    def paras(self):
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._para_elt)

    def lemma_words(self):
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._lemma_word_elt)

    def tagged_words(self, tagset, tags):
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.word_path, MTEFileReader._tagged_word_elt)

    def lemma_sents(self):
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._lemma_sent_elt)

    def tagged_sents(self, tagset, tags):
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.sent_path, MTEFileReader._tagged_sent_elt)

    def lemma_paras(self):
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._lemma_para_elt)

    def tagged_paras(self, tagset, tags):
        MTEFileReader.__tagset = tagset
        MTEFileReader.__tags = tags
        return MTECorpusView(self.__file_path, MTEFileReader.para_path, MTEFileReader._tagged_para_elt)