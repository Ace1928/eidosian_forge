import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
class TestBaseDirectory(unittest.TestCase):

    def test_accepts_valid_dir_name(self):
        foo = ExistsBaseDirectory()
        tempdir = gettempdir()
        self.assertIsInstance(tempdir, str)
        foo.path = tempdir

    def test_rejects_invalid_dir_name(self):
        foo = ExistsBaseDirectory()
        with self.assertRaises(TraitError):
            foo.path = '!!!invalid_directory'

    def test_rejects_valid_file_name(self):
        foo = ExistsBaseDirectory()
        with self.assertRaises(TraitError):
            foo.path = __file__

    def test_accepts_valid_pathlib_dir(self):
        foo = ExistsBaseDirectory()
        foo.path = pathlib.Path(gettempdir())
        self.assertIsInstance(foo.path, str)

    def test_rejects_invalid_pathlib_dir(self):
        foo = ExistsBaseDirectory()
        with self.assertRaises(TraitError):
            foo.path = pathlib.Path('!!!invalid_directory')

    def test_rejects_valid_pathlib_file(self):
        foo = ExistsBaseDirectory()
        with self.assertRaises(TraitError):
            foo.path = pathlib.Path(__file__)

    def test_rejects_invalid_type(self):
        """ Rejects instances that are not `str` or `os.PathLike`.
        """
        foo = ExistsBaseDirectory()
        with self.assertRaises(TraitError):
            foo.path = 1
        with self.assertRaises(TraitError):
            foo.path = b'!!!invalid_directory'

    def test_simple_accepts_any_name(self):
        """ BaseDirectory with no existence check accepts any path name.
        """
        foo = SimpleBaseDirectory()
        foo.path = '!!!invalid_directory'

    def test_simple_accepts_any_pathlib(self):
        """ BaseDirectory with no existence check accepts any pathlib path.
        """
        foo = SimpleBaseDirectory()
        foo.path = pathlib.Path('!!!')
        self.assertIsInstance(foo.path, str)

    def test_info_text(self):
        example_model = ExampleModel()
        with self.assertRaises(TraitError) as exc_cm:
            example_model.path = 47
        self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
        self.assertIn('referring to an existing directory', str(exc_cm.exception))
        with self.assertRaises(TraitError) as exc_cm:
            example_model.new_path = 47
        self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
        self.assertNotIn('exist', str(exc_cm.exception))