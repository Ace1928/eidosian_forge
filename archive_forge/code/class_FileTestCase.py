import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
class FileTestCase(unittest.TestCase):

    def test_valid_file(self):
        example_model = ExampleModel(file_name=__file__)
        example_model.file_name = os.path.__file__

    def test_valid_pathlike_file(self):
        ExampleModel(file_name=Path(__file__))

    def test_invalid_file(self):
        example_model = ExampleModel(file_name=__file__)
        with self.assertRaises(TraitError):
            example_model.file_name = 'not_valid_path!#!#!#'

    def test_invalid_pathlike_file(self):
        example_model = ExampleModel(file_name=__file__)
        with self.assertRaises(TraitError):
            example_model.file_name = Path('not_valid_path!#!#!#')

    def test_directory(self):
        example_model = ExampleModel(file_name=__file__)
        with self.assertRaises(TraitError):
            example_model.file_name = os.path.dirname(__file__)

    def test_pathlike_directory(self):
        example_model = ExampleModel(file_name=__file__)
        with self.assertRaises(TraitError):
            example_model.file_name = Path(os.path.dirname(__file__))

    def test_invalid_type(self):
        example_model = ExampleModel(file_name=__file__)
        with self.assertRaises(TraitError):
            example_model.file_name = 11

    def test_fast(self):
        example_model = FastExampleModel(file_name=__file__)
        example_model.path = '.'

    def test_info_text(self):
        example_model = ExampleModel()
        with self.assertRaises(TraitError) as exc_cm:
            example_model.file_name = 47
        self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
        self.assertIn('referring to an existing file', str(exc_cm.exception))
        with self.assertRaises(TraitError) as exc_cm:
            example_model.new_file_name = 47
        self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
        self.assertNotIn('exist', str(exc_cm.exception))