import os
import zstandard
import ujson as json
import time
import tarfile
import codecs
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path
class JSONArchive:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.data = []
        self.i = 0
        if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
            self.i = max(map(lambda x: int(x.split('_')[1].split('.')[0]), os.listdir(out_dir))) + 1

    def add_data(self, data):
        self.data.append(data)

    def commit(self):
        cctx = zstandard.ZstdCompressor(level=3)
        cdata = cctx.compress(json.dumps(self.data).encode('UTF-8'))
        with open(self.out_dir + '/data_' + str(self.i) + '_' + str(int(time.time())) + '.json.zst', 'wb') as fh:
            fh.write(cdata)
        self.i += 1
        self.data = []