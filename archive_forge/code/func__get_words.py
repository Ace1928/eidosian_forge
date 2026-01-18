import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
def _get_words(self, fileid, speaker, sent, stem, relation, pos, strip_space, replace):
    if isinstance(speaker, str) and speaker != 'ALL':
        speaker = [speaker]
    xmldoc = ElementTree.parse(fileid).getroot()
    results = []
    for xmlsent in xmldoc.findall('.//{%s}u' % NS):
        sents = []
        if speaker == 'ALL' or xmlsent.get('who') in speaker:
            for xmlword in xmlsent.findall('.//{%s}w' % NS):
                infl = None
                suffixStem = None
                suffixTag = None
                if replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement'):
                    xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement/{{{NS}}}w')
                elif replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk'):
                    xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk')
                if xmlword.text:
                    word = xmlword.text
                else:
                    word = ''
                if strip_space:
                    word = word.strip()
                if relation or stem:
                    try:
                        xmlstem = xmlword.find('.//{%s}stem' % NS)
                        word = xmlstem.text
                    except AttributeError as e:
                        pass
                    try:
                        xmlinfl = xmlword.find(f'.//{{{NS}}}mor/{{{NS}}}mw/{{{NS}}}mk')
                        word += '-' + xmlinfl.text
                    except:
                        pass
                    try:
                        xmlsuffix = xmlword.find('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}stem' % (NS, NS, NS, NS))
                        suffixStem = xmlsuffix.text
                    except AttributeError:
                        suffixStem = ''
                    if suffixStem:
                        word += '~' + suffixStem
                if relation or pos:
                    try:
                        xmlpos = xmlword.findall('.//{%s}c' % NS)
                        xmlpos2 = xmlword.findall('.//{%s}s' % NS)
                        if xmlpos2 != []:
                            tag = xmlpos[0].text + ':' + xmlpos2[0].text
                        else:
                            tag = xmlpos[0].text
                    except (AttributeError, IndexError) as e:
                        tag = ''
                    try:
                        xmlsuffixpos = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}c' % (NS, NS, NS, NS, NS))
                        xmlsuffixpos2 = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}s' % (NS, NS, NS, NS, NS))
                        if xmlsuffixpos2:
                            suffixTag = xmlsuffixpos[0].text + ':' + xmlsuffixpos2[0].text
                        else:
                            suffixTag = xmlsuffixpos[0].text
                    except:
                        pass
                    if suffixTag:
                        tag += '~' + suffixTag
                    word = (word, tag)
                if relation == True:
                    for xmlstem_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}gra'):
                        if not xmlstem_rel.get('type') == 'grt':
                            word = (word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                        else:
                            word = (word[0], word[1], word[2], word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                    try:
                        for xmlpost_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}mor-post/{{{NS}}}gra'):
                            if not xmlpost_rel.get('type') == 'grt':
                                suffixStem = (suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                            else:
                                suffixStem = (suffixStem[0], suffixStem[1], suffixStem[2], suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                    except:
                        pass
                sents.append(word)
            if sent or relation:
                results.append(sents)
            else:
                results.extend(sents)
    return LazyMap(lambda x: x, results)