import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def _prepare_images(self):
    if hasattr(self, '_image_files'):
        return
    self._image_files = {}
    for filename in os.listdir(asset(ASSETDIR)):
        matches = v0_FILE_RE.match(filename)
        if not matches:
            continue
        w, h, pat, alpha, fmtinfo, tst, encoder, ext = matches.groups()
        self._image_files[filename] = {'filename': filename, 'w': int(w), 'h': int(h), 'pattern': pat, 'alpha': alpha, 'fmtinfo': fmtinfo, 'testname': tst, 'encoder': encoder, 'ext': ext, 'require_alpha': 'BINARY' in tst or 'ALPHA' in tst}