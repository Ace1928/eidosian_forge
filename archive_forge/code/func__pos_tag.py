from nltk.tag.api import TaggerI
from nltk.tag.util import str2tuple, tuple2str, untag
from nltk.tag.sequential import (
from nltk.tag.brill import BrillTagger
from nltk.tag.brill_trainer import BrillTaggerTrainer
from nltk.tag.tnt import TnT
from nltk.tag.hunpos import HunposTagger
from nltk.tag.stanford import StanfordTagger, StanfordPOSTagger, StanfordNERTagger
from nltk.tag.hmm import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.tag.senna import SennaTagger, SennaChunkTagger, SennaNERTagger
from nltk.tag.mapping import tagset_mapping, map_tag
from nltk.tag.crf import CRFTagger
from nltk.tag.perceptron import PerceptronTagger
from nltk.data import load, find
def _pos_tag(tokens, tagset=None, tagger=None, lang=None):
    if lang not in ['eng', 'rus']:
        raise NotImplementedError("Currently, NLTK pos_tag only supports English and Russian (i.e. lang='eng' or lang='rus')")
    elif isinstance(tokens, str):
        raise TypeError('tokens: expected a list of strings, got a string')
    else:
        tagged_tokens = tagger.tag(tokens)
        if tagset:
            if lang == 'eng':
                tagged_tokens = [(token, map_tag('en-ptb', tagset, tag)) for token, tag in tagged_tokens]
            elif lang == 'rus':
                tagged_tokens = [(token, map_tag('ru-rnc-new', tagset, tag.partition('=')[0])) for token, tag in tagged_tokens]
        return tagged_tokens