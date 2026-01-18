import os
import tempfile
import warnings
from abc import abstractmethod
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_file, find_jar, java
from nltk.tag.api import TaggerI
class StanfordNERTagger(StanfordTagger):
    """
    A class for Named-Entity Tagging with Stanford Tagger. The input is the paths to:

    - a model trained on training data
    - (optionally) the path to the stanford tagger jar file. If not specified here,
      then this jar file must be specified in the CLASSPATH environment variable.
    - (optionally) the encoding of the training data (default: UTF-8)

    Example:

        >>> from nltk.tag import StanfordNERTagger
        >>> st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz') # doctest: +SKIP
        >>> st.tag('Rami Eid is studying at Stony Brook University in NY'.split()) # doctest: +SKIP
        [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'),
         ('at', 'O'), ('Stony', 'ORGANIZATION'), ('Brook', 'ORGANIZATION'),
         ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'LOCATION')]
    """
    _SEPARATOR = '/'
    _JAR = 'stanford-ner.jar'
    _FORMAT = 'slashTags'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _cmd(self):
        return ['edu.stanford.nlp.ie.crf.CRFClassifier', '-loadClassifier', self._stanford_model, '-textFile', self._input_file_path, '-outputFormat', self._FORMAT, '-tokenizerFactory', 'edu.stanford.nlp.process.WhitespaceTokenizer', '-tokenizerOptions', '"tokenizeNLs=false"']

    def parse_output(self, text, sentences):
        if self._FORMAT == 'slashTags':
            tagged_sentences = []
            for tagged_sentence in text.strip().split('\n'):
                for tagged_word in tagged_sentence.strip().split():
                    word_tags = tagged_word.strip().split(self._SEPARATOR)
                    tagged_sentences.append((''.join(word_tags[:-1]), word_tags[-1]))
            result = []
            start = 0
            for sent in sentences:
                result.append(tagged_sentences[start:start + len(sent)])
                start += len(sent)
            return result
        raise NotImplementedError