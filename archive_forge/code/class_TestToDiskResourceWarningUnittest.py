import warnings
from unittest import TestCase
import pytest
import srsly
from numpy import zeros
from spacy.kb.kb_in_memory import InMemoryLookupKB, Writer
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
class TestToDiskResourceWarningUnittest(TestCase):

    def test_resource_warning(self):
        scenarios = zip(*objects_to_test)
        for scenario in scenarios:
            with self.subTest(msg=scenario[1]):
                warnings_list = write_obj_and_catch_warnings(scenario[0])
                self.assertEqual(len(warnings_list), 0)