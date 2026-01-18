import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag, str2tuple
def _parse_utterance(self, utterance, include_tag, tagset=None):
    m = self._UTTERANCE_RE.match(utterance)
    if m is None:
        raise ValueError('Bad utterance %r' % utterance)
    speaker, id, text = m.groups()
    words = [str2tuple(s, self._SEP) for s in text.split()]
    if not include_tag:
        words = [w for w, t in words]
    elif tagset and tagset != self._tagset:
        words = [(w, map_tag(self._tagset, tagset, t)) for w, t in words]
    return SwitchboardTurn(words, speaker, id)