import os
import tempfile
import warnings
from abc import abstractmethod
from subprocess import PIPE
from nltk.internals import _java_options, config_java, find_file, find_jar, java
from nltk.tag.api import TaggerI
class StanfordTagger(TaggerI):
    """
    An interface to Stanford taggers. Subclasses must define:

    - ``_cmd`` property: A property that returns the command that will be
      executed.
    - ``_SEPARATOR``: Class constant that represents that character that
      is used to separate the tokens from their tags.
    - ``_JAR`` file: Class constant that represents the jar file name.
    """
    _SEPARATOR = ''
    _JAR = ''

    def __init__(self, model_filename, path_to_jar=None, encoding='utf8', verbose=False, java_options='-mx1000m'):
        warnings.warn(str('\nThe StanfordTokenizer will be deprecated in version 3.2.6.\nPlease use \x1b[91mnltk.parse.corenlp.CoreNLPParser\x1b[0m instead.'), DeprecationWarning, stacklevel=2)
        if not self._JAR:
            warnings.warn('The StanfordTagger class is not meant to be instantiated directly. Did you mean StanfordPOSTagger or StanfordNERTagger?')
        self._stanford_jar = find_jar(self._JAR, path_to_jar, searchpath=(), url=_stanford_url, verbose=verbose)
        self._stanford_model = find_file(model_filename, env_vars=('STANFORD_MODELS',), verbose=verbose)
        self._encoding = encoding
        self.java_options = java_options

    @property
    @abstractmethod
    def _cmd(self):
        """
        A property that returns the command that will be executed.
        """

    def tag(self, tokens):
        return sum(self.tag_sents([tokens]), [])

    def tag_sents(self, sentences):
        encoding = self._encoding
        default_options = ' '.join(_java_options)
        config_java(options=self.java_options, verbose=False)
        _input_fh, self._input_file_path = tempfile.mkstemp(text=True)
        cmd = list(self._cmd)
        cmd.extend(['-encoding', encoding])
        _input_fh = os.fdopen(_input_fh, 'wb')
        _input = '\n'.join((' '.join(x) for x in sentences))
        if isinstance(_input, str) and encoding:
            _input = _input.encode(encoding)
        _input_fh.write(_input)
        _input_fh.close()
        stanpos_output, _stderr = java(cmd, classpath=self._stanford_jar, stdout=PIPE, stderr=PIPE)
        stanpos_output = stanpos_output.decode(encoding)
        os.unlink(self._input_file_path)
        config_java(options=default_options, verbose=False)
        return self.parse_output(stanpos_output, sentences)

    def parse_output(self, text, sentences=None):
        tagged_sentences = []
        for tagged_sentence in text.strip().split('\n'):
            sentence = []
            for tagged_word in tagged_sentence.strip().split():
                word_tags = tagged_word.strip().split(self._SEPARATOR)
                sentence.append((''.join(word_tags[:-1]), word_tags[-1].replace('0', '').upper()))
            tagged_sentences.append(sentence)
        return tagged_sentences