import pickle
import threading
from .. import errors, osutils, tests
from ..tests import features
def _temppath(self, ext):
    return osutils.pathjoin(self.test_dir, 'tmp_profile_data.' + ext)