import os
import re
from functools import reduce
from nltk.corpus.reader import TaggedCorpusReader, concat
from nltk.corpus.reader.xmldocs import XMLCorpusView
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