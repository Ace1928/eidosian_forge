import json
from struct import pack
from pprint import pprint
from subprocess import Popen
from PIL import Image
from argparse import ArgumentParser
from sys import exit
from os.path import join, exists, dirname, basename
from os import environ, unlink
class PvrtcTool(Tool):

    def __init__(self, options):
        super(PvrtcTool, self).__init__(options)
        self.texturetool = None
        self.locate_texturetool()

    def locate_texturetool(self):
        search_directories = ['/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/', '/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/']
        search_directories += environ.get('PATH', '').split(':')
        for directory in search_directories:
            fn = join(directory, 'texturetool')
            if not exists(fn):
                continue
            print('Found texturetool at {}'.format(directory))
            self.texturetool = fn
            return
        print('Error: Unable to locate "texturetool".\nPlease install the iPhone SDK, or the PowerVR SDK.\nThen make sure that "texturetool" is available in your PATH.')
        exit(1)

    def compress(self):
        image = Image.open(self.source_fn)
        w, h = image.size
        print('Image size is {}x{}'.format(*image.size))
        w2 = self.nearest_pow2(w)
        h2 = self.nearest_pow2(h)
        print('Nearest power-of-2 size is {}x{}'.format(w2, h2))
        s2 = max(w2, h2)
        print('PVR need a square image, the texture will be {0}x{0}'.format(s2))
        ext = self.source_fn.rsplit('.', 1)[-1]
        tmpfile = '/tmp/ktexturecompress.{}'.format(ext)
        image = image.resize((s2, s2))
        image.save(tmpfile)
        raw_tex_fn = self.tex_fn + '.raw'
        cmd = [self.texturetool]
        if self.options.mipmap:
            cmd += ['-m']
        cmd += ['-e', 'PVRTC', '-o', raw_tex_fn, '-f', 'RAW', tmpfile]
        try:
            self.runcmd(cmd)
            with open(raw_tex_fn, 'rb') as fd:
                data = fd.read()
        finally:
            if exists(raw_tex_fn):
                unlink(raw_tex_fn)
        self.write_tex(data, 'pvrtc_rgba4', (w, h), (s2, s2), self.options.mipmap)