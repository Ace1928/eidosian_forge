import abc
import hashlib
import os
import tempfile
from pathlib import Path
from ..common.build import _build
from .cache import get_cache_manager
class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src = Path(os.path.join(dirname, 'backends', 'cuda.c')).read_text()
        key = hashlib.md5(src.encode('utf-8')).hexdigest()
        cache = get_cache_manager(key)
        fname = 'cuda_utils.so'
        cache_path = cache.get_file(fname)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src_path = os.path.join(tmpdir, 'main.c')
                with open(src_path, 'w') as f:
                    f.write(src)
                so = _build('cuda_utils', src_path, tmpdir)
                with open(so, 'rb') as f:
                    cache_path = cache.put(f.read(), fname, binary=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location('cuda_utils', cache_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.load_binary = mod.load_binary
        self.get_device_properties = mod.get_device_properties
        self.CUtensorMapDataType = mod.CUtensorMapDataType
        self.CUtensorMapInterleave = mod.CUtensorMapInterleave
        self.CUtensorMapSwizzle = mod.CUtensorMapSwizzle
        self.CUtensorMapL2promotion = mod.CUtensorMapL2promotion
        self.CUtensorMapFloatOOBfill = mod.CUtensorMapFloatOOBfill
        self.cuTensorMapEncodeTiled = mod.cuTensorMapEncodeTiled
        self.cuMemAlloc = mod.cuMemAlloc
        self.cuMemcpyHtoD = mod.cuMemcpyHtoD
        self.cuMemFree = mod.cuMemFree
        self.cuOccupancyMaxActiveClusters = mod.cuOccupancyMaxActiveClusters