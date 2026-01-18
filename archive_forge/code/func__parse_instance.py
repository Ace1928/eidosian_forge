import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
def _parse_instance(self, instance, lexelt):
    senses = []
    context = []
    position = None
    for child in instance:
        if child.tag == 'answer':
            senses.append(child.attrib['senseid'])
        elif child.tag == 'context':
            context += self._word_tokenizer.tokenize(child.text)
            for cword in child:
                if cword.tag == 'compound':
                    cword = cword[0]
                if cword.tag == 'head':
                    assert position is None, 'head specified twice'
                    assert cword.text.strip() or len(cword) == 1
                    assert not (cword.text.strip() and len(cword) == 1)
                    position = len(context)
                    if cword.text.strip():
                        context.append(cword.text.strip())
                    elif cword[0].tag == 'wf':
                        context.append((cword[0].text, cword[0].attrib['pos']))
                        if cword[0].tail:
                            context += self._word_tokenizer.tokenize(cword[0].tail)
                    else:
                        assert False, 'expected CDATA or wf in <head>'
                elif cword.tag == 'wf':
                    context.append((cword.text, cword.attrib['pos']))
                elif cword.tag == 's':
                    pass
                else:
                    print('ACK', cword.tag)
                    assert False, 'expected CDATA or <wf> or <head>'
                if cword.tail:
                    context += self._word_tokenizer.tokenize(cword.tail)
        else:
            assert False, 'unexpected tag %s' % child.tag
    return SensevalInstance(lexelt, position, context, senses)