import shutil
import glob
import os
import sys
import tempfile
class AppsiBuilder(object):

    def __call__(self, parallel):
        return build_appsi()