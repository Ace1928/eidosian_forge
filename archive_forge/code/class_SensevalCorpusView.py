import re
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
class SensevalCorpusView(StreamBackedCorpusView):

    def __init__(self, fileid, encoding):
        StreamBackedCorpusView.__init__(self, fileid, encoding=encoding)
        self._word_tokenizer = WhitespaceTokenizer()
        self._lexelt_starts = [0]
        self._lexelts = [None]

    def read_block(self, stream):
        lexelt_num = bisect.bisect_right(self._lexelt_starts, stream.tell()) - 1
        lexelt = self._lexelts[lexelt_num]
        instance_lines = []
        in_instance = False
        while True:
            line = stream.readline()
            if line == '':
                assert instance_lines == []
                return []
            if line.lstrip().startswith('<lexelt'):
                lexelt_num += 1
                m = re.search('item=("[^"]+"|\'[^\']+\')', line)
                assert m is not None
                lexelt = m.group(1)[1:-1]
                if lexelt_num < len(self._lexelts):
                    assert lexelt == self._lexelts[lexelt_num]
                else:
                    self._lexelts.append(lexelt)
                    self._lexelt_starts.append(stream.tell())
            if line.lstrip().startswith('<instance'):
                assert instance_lines == []
                in_instance = True
            if in_instance:
                instance_lines.append(line)
            if line.lstrip().startswith('</instance'):
                xml_block = '\n'.join(instance_lines)
                xml_block = _fixXML(xml_block)
                inst = ElementTree.fromstring(xml_block)
                return [self._parse_instance(inst, lexelt)]

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