import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def _test_imageloader(self, loadercls, extensions=None):
    if not loadercls:
        return
    if not extensions:
        extensions = loadercls.extensions()
    ctx = _TestContext(loadercls)
    self._context = ctx
    for filename in sorted(self._image_files.keys()):
        filedata = self._image_files[filename]
        if filedata['ext'] not in extensions:
            continue
        try:
            ctx.start(filename, filedata)
            result = loadercls(asset(ASSETDIR, filename), keep_data=True)
            if not result:
                raise Exception('invalid result')
        except:
            ctx.skip('Error loading file, result=None')
            continue
        self._test_image(filedata, ctx, loadercls, result)
        ctx.end()
    ok, skip, fail, stats = ctx.results
    if fail:
        self.fail('{}: {} passed, {} skipped, {} failed'.format(loadercls.__name__, ok, skip, fail))
    return ctx