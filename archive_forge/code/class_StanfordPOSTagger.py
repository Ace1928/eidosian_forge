import os
import tempfile
import warnings
from abc import abstractmethod
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_file, find_jar, java
from nltk.tag.api import TaggerI
class StanfordPOSTagger(StanfordTagger):
    """
    A class for pos tagging with Stanford Tagger. The input is the paths to:
     - a model trained on training data
     - (optionally) the path to the stanford tagger jar file. If not specified here,
       then this jar file must be specified in the CLASSPATH environment variable.
     - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordPOSTagger
        >>> st = StanfordPOSTagger('english-bidirectional-distsim.tagger') # doctest: +SKIP
        >>> st.tag('What is the airspeed of an unladen swallow ?'.split()) # doctest: +SKIP
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'), ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'), ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
    """
    _SEPARATOR = '_'
    _JAR = 'stanford-postagger.jar'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.tagger.maxent.MaxentTagger', '-model', self._stanford_model, '-textFile', self._input_file_path, '-tokenize', 'false', '-outputFormatOptions', 'keepEmptySentences']