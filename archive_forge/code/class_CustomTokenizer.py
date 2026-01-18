import pytest
from thinc.api import ConfigValidationError
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
class CustomTokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.from_initialize = None

    def __call__(self, text):
        return self.tokenizer(text)

    def initialize(self, get_examples, nlp, custom: int):
        self.from_initialize = custom