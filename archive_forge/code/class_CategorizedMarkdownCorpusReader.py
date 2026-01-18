from collections import namedtuple
from functools import partial, wraps
from nltk.corpus.reader.api import CategorizedCorpusReader
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.util import concat, read_blankline_block
from nltk.tokenize import blankline_tokenize, sent_tokenize, word_tokenize
class CategorizedMarkdownCorpusReader(CategorizedCorpusReader, MarkdownCorpusReader):
    """
    A reader for markdown corpora whose documents are divided into
    categories based on their file identifiers.

    Based on nltk.corpus.reader.plaintext.CategorizedPlaintextCorpusReader:
    https://www.nltk.org/_modules/nltk/corpus/reader/api.html#CategorizedCorpusReader
    """

    def __init__(self, *args, cat_field='tags', **kwargs):
        """
        Initialize the corpus reader. Categorization arguments
        (``cat_pattern``, ``cat_map``, and ``cat_file``) are passed to
        the ``CategorizedCorpusReader`` constructor.  The remaining arguments
        are passed to the ``MarkdownCorpusReader`` constructor.
        """
        cat_args = ['cat_pattern', 'cat_map', 'cat_file']
        if not any((arg in kwargs for arg in cat_args)):
            kwargs['cat_map'] = dict()
        CategorizedCorpusReader.__init__(self, kwargs)
        MarkdownCorpusReader.__init__(self, *args, **kwargs)
        if self._map is not None and (not self._map):
            for file_id in self._fileids:
                metadata = self.metadata(file_id)
                if metadata:
                    self._map[file_id] = metadata[0].get(cat_field, [])

    @comma_separated_string_args
    def categories(self, fileids=None):
        return super().categories(fileids)

    @comma_separated_string_args
    def fileids(self, categories=None):
        if categories is None:
            return self._fileids
        return super().fileids(categories)

    @comma_separated_string_args
    def raw(self, fileids=None, categories=None):
        return super().raw(self._resolve(fileids, categories))

    @comma_separated_string_args
    def words(self, fileids=None, categories=None):
        return super().words(self._resolve(fileids, categories))

    @comma_separated_string_args
    def sents(self, fileids=None, categories=None):
        return super().sents(self._resolve(fileids, categories))

    @comma_separated_string_args
    def paras(self, fileids=None, categories=None):
        return super().paras(self._resolve(fileids, categories))

    def concatenated_view(self, reader, fileids, categories):
        return concat([self.CorpusView(path, reader, encoding=enc) for path, enc in self.abspaths(self._resolve(fileids, categories), include_encoding=True)])

    def metadata_reader(self, stream):
        from yaml import safe_load
        return [safe_load(t.content) for t in self.parser.parse(stream.read()) if t.type == 'front_matter']

    @comma_separated_string_args
    def metadata(self, fileids=None, categories=None):
        return self.concatenated_view(self.metadata_reader, fileids, categories)

    def blockquote_reader(self, stream):
        tokens = self.parser.parse(stream.read())
        opening_tokens = filter(lambda t: t.level == 0 and t.type == 'blockquote_open', tokens)
        closing_tokens = filter(lambda t: t.level == 0 and t.type == 'blockquote_close', tokens)
        blockquotes = list()
        for o, c in zip(opening_tokens, closing_tokens):
            opening_index = tokens.index(o)
            closing_index = tokens.index(c, opening_index)
            blockquotes.append(tokens[opening_index:closing_index + 1])
        return [MarkdownBlock(self.parser.renderer.render(block, self.parser.options, env=None)) for block in blockquotes]

    @comma_separated_string_args
    def blockquotes(self, fileids=None, categories=None):
        return self.concatenated_view(self.blockquote_reader, fileids, categories)

    def code_block_reader(self, stream):
        return [CodeBlock(t.info, t.content) for t in self.parser.parse(stream.read()) if t.level == 0 and t.type in ('fence', 'code_block')]

    @comma_separated_string_args
    def code_blocks(self, fileids=None, categories=None):
        return self.concatenated_view(self.code_block_reader, fileids, categories)

    def image_reader(self, stream):
        return [Image(child_token.content, child_token.attrGet('src'), child_token.attrGet('title')) for inline_token in filter(lambda t: t.type == 'inline', self.parser.parse(stream.read())) for child_token in inline_token.children if child_token.type == 'image']

    @comma_separated_string_args
    def images(self, fileids=None, categories=None):
        return self.concatenated_view(self.image_reader, fileids, categories)

    def link_reader(self, stream):
        return [Link(inline_token.children[i + 1].content, child_token.attrGet('href'), child_token.attrGet('title')) for inline_token in filter(lambda t: t.type == 'inline', self.parser.parse(stream.read())) for i, child_token in enumerate(inline_token.children) if child_token.type == 'link_open']

    @comma_separated_string_args
    def links(self, fileids=None, categories=None):
        return self.concatenated_view(self.link_reader, fileids, categories)

    def list_reader(self, stream):
        tokens = self.parser.parse(stream.read())
        opening_types = ('bullet_list_open', 'ordered_list_open')
        opening_tokens = filter(lambda t: t.level == 0 and t.type in opening_types, tokens)
        closing_types = ('bullet_list_close', 'ordered_list_close')
        closing_tokens = filter(lambda t: t.level == 0 and t.type in closing_types, tokens)
        list_blocks = list()
        for o, c in zip(opening_tokens, closing_tokens):
            opening_index = tokens.index(o)
            closing_index = tokens.index(c, opening_index)
            list_blocks.append(tokens[opening_index:closing_index + 1])
        return [List(tokens[0].type == 'ordered_list_open', [t.content for t in tokens if t.content]) for tokens in list_blocks]

    @comma_separated_string_args
    def lists(self, fileids=None, categories=None):
        return self.concatenated_view(self.list_reader, fileids, categories)

    def section_reader(self, stream):
        section_blocks, block = (list(), list())
        in_heading = False
        for t in self.parser.parse(stream.read()):
            if t.level == 0 and t.type == 'heading_open':
                if block:
                    section_blocks.append(block)
                block = list()
                in_heading = True
            if in_heading:
                block.append(t)
        return [MarkdownSection(block[1].content, block[0].markup.count('#'), self.parser.renderer.render(block, self.parser.options, env=None)) for block in section_blocks]

    @comma_separated_string_args
    def sections(self, fileids=None, categories=None):
        return self.concatenated_view(self.section_reader, fileids, categories)