import os
import tempfile
import unittest
from pathlib import Path
from bpython.importcompletion import ModuleGatherer
class TestAvoidSymbolicLinks(unittest.TestCase):

    def setUp(self):
        with tempfile.TemporaryDirectory() as import_test_folder:
            base_path = Path(import_test_folder)
            (base_path / 'Level0' / 'Level1' / 'Level2').mkdir(parents=True)
            (base_path / 'Left').mkdir(parents=True)
            (base_path / 'Right').mkdir(parents=True)
            current_path = base_path / 'Level0'
            (current_path / '__init__.py').touch()
            current_path = current_path / 'Level1'
            (current_path / '__init__.py').touch()
            current_path = current_path / 'Level2'
            (current_path / '__init__.py').touch()
            (current_path / 'Level3').symlink_to(base_path / 'Level0' / 'Level1', target_is_directory=True)
            current_path = base_path / 'Right'
            (current_path / '__init__.py').touch()
            (current_path / 'toLeft').symlink_to(base_path / 'Left', target_is_directory=True)
            current_path = base_path / 'Left'
            (current_path / '__init__.py').touch()
            (current_path / 'toRight').symlink_to(base_path / 'Right', target_is_directory=True)
            self.module_gatherer = ModuleGatherer((base_path.absolute(),))
            while self.module_gatherer.find_coroutine():
                pass

    def test_simple_symbolic_link_loop(self):
        filepaths = ['Left.toRight.toLeft', 'Left.toRight', 'Left', 'Level0.Level1.Level2.Level3', 'Level0.Level1.Level2', 'Level0.Level1', 'Level0', 'Right', 'Right.toLeft', 'Right.toLeft.toRight']
        for thing in self.module_gatherer.modules:
            self.assertIn(thing, filepaths)
            if thing == 'Left.toRight.toLeft':
                filepaths.remove('Right.toLeft')
                filepaths.remove('Right.toLeft.toRight')
            if thing == 'Right.toLeft.toRight':
                filepaths.remove('Left.toRight.toLeft')
                filepaths.remove('Left.toRight')
            filepaths.remove(thing)
        self.assertFalse(filepaths)