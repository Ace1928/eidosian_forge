import json
import os  # required for doctests
import re
import socket
import time
from typing import List, Tuple
from nltk.internals import _java_options, config_java, find_jar_iter, java
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tag.api import TaggerI
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree
class GenericCoreNLPParser(ParserI, TokenizerI, TaggerI):
    """Interface to the CoreNLP Parser."""

    def __init__(self, url='http://localhost:9000', encoding='utf8', tagtype=None, strict_json=True):
        import requests
        self.url = url
        self.encoding = encoding
        if tagtype not in ['pos', 'ner', None]:
            raise ValueError("tagtype must be either 'pos', 'ner' or None")
        self.tagtype = tagtype
        self.strict_json = strict_json
        self.session = requests.Session()

    def parse_sents(self, sentences, *args, **kwargs):
        """Parse multiple sentences.

        Takes multiple sentences as a list where each sentence is a list of
        words. Each sentence will be automatically tagged with this
        CoreNLPParser instance's tagger.

        If a whitespace exists inside a token, then the token will be treated as
        several tokens.

        :param sentences: Input sentences to parse
        :type sentences: list(list(str))
        :rtype: iter(iter(Tree))
        """
        sentences = (' '.join(words) for words in sentences)
        return self.raw_parse_sents(sentences, *args, **kwargs)

    def raw_parse(self, sentence, properties=None, *args, **kwargs):
        """Parse a sentence.

        Takes a sentence as a string; before parsing, it will be automatically
        tokenized and tagged by the CoreNLP Parser.

        :param sentence: Input sentence to parse
        :type sentence: str
        :rtype: iter(Tree)
        """
        default_properties = {'tokenize.whitespace': 'false'}
        default_properties.update(properties or {})
        return next(self.raw_parse_sents([sentence], *args, properties=default_properties, **kwargs))

    def api_call(self, data, properties=None, timeout=60):
        default_properties = {'outputFormat': 'json', 'annotators': 'tokenize,pos,lemma,ssplit,{parser_annotator}'.format(parser_annotator=self.parser_annotator)}
        default_properties.update(properties or {})
        response = self.session.post(self.url, params={'properties': json.dumps(default_properties)}, data=data.encode(self.encoding), headers={'Content-Type': f'text/plain; charset={self.encoding}'}, timeout=timeout)
        response.raise_for_status()
        return response.json(strict=self.strict_json)

    def raw_parse_sents(self, sentences, verbose=False, properties=None, *args, **kwargs):
        """Parse multiple sentences.

        Takes multiple sentences as a list of strings. Each sentence will be
        automatically tokenized and tagged.

        :param sentences: Input sentences to parse.
        :type sentences: list(str)
        :rtype: iter(iter(Tree))

        """
        default_properties = {'ssplit.eolonly': 'true'}
        default_properties.update(properties or {})
        "\n        for sentence in sentences:\n            parsed_data = self.api_call(sentence, properties=default_properties)\n\n            assert len(parsed_data['sentences']) == 1\n\n            for parse in parsed_data['sentences']:\n                tree = self.make_tree(parse)\n                yield iter([tree])\n        "
        parsed_data = self.api_call('\n'.join(sentences), properties=default_properties)
        for parsed_sent in parsed_data['sentences']:
            tree = self.make_tree(parsed_sent)
            yield iter([tree])

    def parse_text(self, text, *args, **kwargs):
        """Parse a piece of text.

        The text might contain several sentences which will be split by CoreNLP.

        :param str text: text to be split.
        :returns: an iterable of syntactic structures.  # TODO: should it be an iterable of iterables?

        """
        parsed_data = self.api_call(text, *args, **kwargs)
        for parse in parsed_data['sentences']:
            yield self.make_tree(parse)

    def tokenize(self, text, properties=None):
        """Tokenize a string of text.

        Skip these tests if CoreNLP is likely not ready.
        >>> from nltk.test.setup_fixt import check_jar
        >>> check_jar(CoreNLPServer._JAR, env_vars=("CORENLP",), is_regex=True)

        The CoreNLP server can be started using the following notation, although
        we recommend the `with CoreNLPServer() as server:` context manager notation
        to ensure that the server is always stopped.
        >>> server = CoreNLPServer()
        >>> server.start()
        >>> parser = CoreNLPParser(url=server.url)

        >>> text = 'Good muffins cost $3.88\\nin New York.  Please buy me\\ntwo of them.\\nThanks.'
        >>> list(parser.tokenize(text))
        ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York', '.', 'Please', 'buy', 'me', 'two', 'of', 'them', '.', 'Thanks', '.']

        >>> s = "The colour of the wall is blue."
        >>> list(
        ...     parser.tokenize(
        ...         'The colour of the wall is blue.',
        ...             properties={'tokenize.options': 'americanize=true'},
        ...     )
        ... )
        ['The', 'colour', 'of', 'the', 'wall', 'is', 'blue', '.']
        >>> server.stop()

        """
        default_properties = {'annotators': 'tokenize,ssplit'}
        default_properties.update(properties or {})
        result = self.api_call(text, properties=default_properties)
        for sentence in result['sentences']:
            for token in sentence['tokens']:
                yield (token['originalText'] or token['word'])

    def tag_sents(self, sentences):
        """
        Tag multiple sentences.

        Takes multiple sentences as a list where each sentence is a list of
        tokens.

        :param sentences: Input sentences to tag
        :type sentences: list(list(str))
        :rtype: list(list(tuple(str, str))
        """
        sentences = (' '.join(words) for words in sentences)
        return [sentences[0] for sentences in self.raw_tag_sents(sentences)]

    def tag(self, sentence: str) -> List[Tuple[str, str]]:
        """
        Tag a list of tokens.

        :rtype: list(tuple(str, str))

        Skip these tests if CoreNLP is likely not ready.
        >>> from nltk.test.setup_fixt import check_jar
        >>> check_jar(CoreNLPServer._JAR, env_vars=("CORENLP",), is_regex=True)

        The CoreNLP server can be started using the following notation, although
        we recommend the `with CoreNLPServer() as server:` context manager notation
        to ensure that the server is always stopped.
        >>> server = CoreNLPServer()
        >>> server.start()
        >>> parser = CoreNLPParser(url=server.url, tagtype='ner')
        >>> tokens = 'Rami Eid is studying at Stony Brook University in NY'.split()
        >>> parser.tag(tokens)  # doctest: +NORMALIZE_WHITESPACE
        [('Rami', 'PERSON'), ('Eid', 'PERSON'), ('is', 'O'), ('studying', 'O'), ('at', 'O'), ('Stony', 'ORGANIZATION'),
        ('Brook', 'ORGANIZATION'), ('University', 'ORGANIZATION'), ('in', 'O'), ('NY', 'STATE_OR_PROVINCE')]

        >>> parser = CoreNLPParser(url=server.url, tagtype='pos')
        >>> tokens = "What is the airspeed of an unladen swallow ?".split()
        >>> parser.tag(tokens)  # doctest: +NORMALIZE_WHITESPACE
        [('What', 'WP'), ('is', 'VBZ'), ('the', 'DT'),
        ('airspeed', 'NN'), ('of', 'IN'), ('an', 'DT'),
        ('unladen', 'JJ'), ('swallow', 'VB'), ('?', '.')]
        >>> server.stop()
        """
        return self.tag_sents([sentence])[0]

    def raw_tag_sents(self, sentences):
        """
        Tag multiple sentences.

        Takes multiple sentences as a list where each sentence is a string.

        :param sentences: Input sentences to tag
        :type sentences: list(str)
        :rtype: list(list(list(tuple(str, str)))
        """
        default_properties = {'ssplit.isOneSentence': 'true', 'annotators': 'tokenize,ssplit,'}
        assert self.tagtype in ['pos', 'ner']
        default_properties['annotators'] += self.tagtype
        for sentence in sentences:
            tagged_data = self.api_call(sentence, properties=default_properties)
            yield [[(token['word'], token[self.tagtype]) for token in tagged_sentence['tokens']] for tagged_sentence in tagged_data['sentences']]