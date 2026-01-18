import pytest
from thinc.api import Adam, fix_random_seed
from spacy import registry
from spacy.attrs import NORM
from spacy.language import Language
from spacy.pipeline import DependencyParser, EntityRecognizer
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc
from spacy.training import Example
from spacy.vocab import Vocab
def _ner_example(ner):
    doc = Doc(ner.vocab, words=['Joe', 'loves', 'visiting', 'London', 'during', 'the', 'weekend'])
    gold = {'entities': [(0, 3, 'PERSON'), (19, 25, 'LOC')]}
    return Example.from_dict(doc, gold)