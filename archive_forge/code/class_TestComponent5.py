import pytest
from mock import Mock
from spacy.language import Language
from spacy.pipe_analysis import get_attr_info, validate_attrs
class TestComponent5:

    def __call__(self, doc):
        return doc