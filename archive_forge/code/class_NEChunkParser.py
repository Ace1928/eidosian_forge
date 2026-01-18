import os
import pickle
import re
from xml.etree import ElementTree as ET
from nltk.tag import ClassifierBasedTagger, pos_tag
from nltk.chunk.api import ChunkParserI
from nltk.chunk.util import ChunkScore
from nltk.data import find
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
class NEChunkParser(ChunkParserI):
    """
    Expected input: list of pos-tagged words
    """

    def __init__(self, train):
        self._train(train)

    def parse(self, tokens):
        """
        Each token should be a pos-tagged word
        """
        tagged = self._tagger.tag(tokens)
        tree = self._tagged_to_parse(tagged)
        return tree

    def _train(self, corpus):
        corpus = [self._parse_to_tagged(s) for s in corpus]
        self._tagger = NEChunkParserTagger(train=corpus)

    def _tagged_to_parse(self, tagged_tokens):
        """
        Convert a list of tagged tokens to a chunk-parse tree.
        """
        sent = Tree('S', [])
        for tok, tag in tagged_tokens:
            if tag == 'O':
                sent.append(tok)
            elif tag.startswith('B-'):
                sent.append(Tree(tag[2:], [tok]))
            elif tag.startswith('I-'):
                if sent and isinstance(sent[-1], Tree) and (sent[-1].label() == tag[2:]):
                    sent[-1].append(tok)
                else:
                    sent.append(Tree(tag[2:], [tok]))
        return sent

    @staticmethod
    def _parse_to_tagged(sent):
        """
        Convert a chunk-parse tree to a list of tagged tokens.
        """
        toks = []
        for child in sent:
            if isinstance(child, Tree):
                if len(child) == 0:
                    print('Warning -- empty chunk in sentence')
                    continue
                toks.append((child[0], f'B-{child.label()}'))
                for tok in child[1:]:
                    toks.append((tok, f'I-{child.label()}'))
            else:
                toks.append((child, 'O'))
        return toks