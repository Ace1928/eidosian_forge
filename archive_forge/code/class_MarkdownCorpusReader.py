from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
class MarkdownCorpusReader(PlaintextCorpusReader):

    def __init__(self, *args, parser=None, **kwargs):
        from markdown_it import MarkdownIt
        from mdit_plain.renderer import RendererPlain
        from mdit_py_plugins.front_matter import front_matter_plugin
        self.parser = parser
        if self.parser is None:
            self.parser = MarkdownIt('commonmark', renderer_cls=RendererPlain)
            self.parser.use(front_matter_plugin)
        kwargs.setdefault('para_block_reader', partial(read_parse_blankline_block, parser=self.parser))
        super().__init__(*args, **kwargs)

    def _read_word_block(self, stream):
        words = list()
        for para in self._para_block_reader(stream):
            words.extend(self._word_tokenizer.tokenize(para))
        return words