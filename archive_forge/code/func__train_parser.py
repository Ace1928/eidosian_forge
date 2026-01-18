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
def _train_parser(parser):
    fix_random_seed(1)
    parser.add_label('left')
    parser.initialize(lambda: [_parser_example(parser)])
    sgd = Adam(0.001)
    for i in range(5):
        losses = {}
        doc = Doc(parser.vocab, words=['a', 'b', 'c', 'd'])
        gold = {'heads': [1, 1, 3, 3], 'deps': ['left', 'ROOT', 'left', 'ROOT']}
        example = Example.from_dict(doc, gold)
        parser.update([example], sgd=sgd, losses=losses)
    return parser