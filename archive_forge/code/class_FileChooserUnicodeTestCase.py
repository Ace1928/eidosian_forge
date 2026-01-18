from os import remove, rmdir, mkdir
from os.path import join, dirname, isdir
import unittest
from zipfile import ZipFile
import pytest
from kivy.clock import Clock
from kivy.uix.filechooser import FileChooserListView
from kivy.utils import platform
class FileChooserUnicodeTestCase(unittest.TestCase):

    def setUp(self):
        basepath = dirname(__file__)
        subdir = join(basepath, 'filechooser_files')
        self.subdir = subdir
        ufiles = ['कीवीtestu', 'कीवीtestu' + unicode_char(61166), 'कीवीtestu' + unicode_char(61166 - 1), 'कीवीtestu' + unicode_char(238)]
        self.files = [join(subdir, f) for f in ufiles]
        if not isdir(subdir):
            mkdir(subdir)
        for f in self.files:
            open(f, 'wb').close()
        existfiles = ['à¤•à¥€à¤µà¥€test', 'à¤•à¥€à¤’µà¥€test', 'Ã\xa0Â¤â€¢Ã\xa0Â¥â‚¬Ã\xa0Â¤ÂµÃ\xa0Â¥â‚¬test', 'testl\ufffe', 'testl\uffff']
        self.existfiles = [join(subdir, f) for f in existfiles]
        with ZipFile(join(basepath, 'unicode_files.zip'), 'r') as myzip:
            myzip.extractall(path=subdir)
        for f in self.existfiles:
            open(f, 'rb').close()

    @pytest.fixture(autouse=True)
    def set_clock(self, kivy_clock):
        self.kivy_clock = kivy_clock

    @pytest.mark.skipif(platform == 'macosx' or platform == 'ios', reason='Unicode files unpredictable on MacOS and iOS')
    def test_filechooserlistview_unicode(self):
        wid = FileChooserListView(path=self.subdir)
        Clock.tick()
        files = [join(self.subdir, f) for f in wid.files]
        for f in self.files:
            self.assertIn(f, files)
        for f in self.existfiles:
            self.assertIn(f, files)

    def tearDown(self):
        for f in self.files + self.existfiles:
            try:
                remove(f)
            except (OSError, FileNotFoundError):
                pass
        try:
            rmdir(self.subdir)
        except (OSError, FileNotFoundError):
            pass