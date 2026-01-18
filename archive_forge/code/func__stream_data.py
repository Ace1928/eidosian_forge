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
def _stream_data(self, get_meta=False, jsonl_key='text'):
    self.f_name = ''
    files = listdir_or_file(self.in_path)
    if not files:
        raise FileNotFoundError(f'No valid file(s) found in {self.in_path}')
    for f in files:
        self.f_name = f
        if f == 'openwebtext.tar.xz':
            assert not get_meta
            yield from self.read_owt(f)
        elif 'urlsf_subset' in f and f.endswith('_data.xz'):
            assert not get_meta
            yield from self.read_owt_subset(f)
        elif f.endswith('.dat.zst'):
            assert not get_meta
            yield from self.read_dat(f)
        elif f.endswith('.jsonl'):
            yield from self.read_jsonl(f, get_meta, key=jsonl_key)
        elif f.endswith('.jsonl.zst'):
            yield from self.read_jsonl_zst(f, get_meta, key=jsonl_key)
        elif f.endswith('.jsonl.zst.tar'):
            yield from self.read_jsonl_tar(f, get_meta, jsonl_key=key)
        elif f.endswith('.json.zst'):
            assert not get_meta
            yield from self.read_json(f)
        elif f.endswith('.txt'):
            assert not get_meta
            yield from self.read_txt(f)
        elif f.endswith('.zip'):
            assert not get_meta
            yield from self.read_zip(f)
        elif f.endswith('.tar.gz'):
            assert not get_meta
            yield from self.read_tgz(f)
        elif f.endswith('.json.gz'):
            assert not get_meta
            yield from self.read_jsongz(f)
        elif f.endswith('.gz'):
            assert not get_meta
            yield from self.read_gz(f)
        else:
            print(f'Skipping {f} as streaming for that filetype is not implemented')