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
def _get_tagger(lang=None):
    if lang == 'rus':
        tagger = PerceptronTagger(False)
        ap_russian_model_loc = 'file:' + str(find(RUS_PICKLE))
        tagger.load(ap_russian_model_loc)
    else:
        tagger = PerceptronTagger()
    return tagger