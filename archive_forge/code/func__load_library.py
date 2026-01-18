import os
import ctypes
import sys
import inspect
import warnings
import types
import tensorflow as tf
def _load_library(filename):
    """_load_library"""
    f = inspect.getfile(sys._getframe(1))
    f = os.path.join(os.path.dirname(f), filename)
    filenames = [f]
    datapath = os.environ.get('TFIO_DATAPATH')
    if datapath is not None:
        rootpath = os.path.dirname(sys.modules['tensorflow_io_gcs_filesystem'].__file__)
        filename = sys.modules[__name__].__file__
        f = os.path.join(datapath, 'tensorflow_io_gcs_filesystem', os.path.relpath(os.path.dirname(filename), rootpath), os.path.relpath(f, os.path.dirname(filename)))
        filenames.append(f)
    load_fn = lambda f: tf.experimental.register_filesystem_plugin(f) is None
    errs = []
    for f in filenames:
        try:
            l = load_fn(f)
            if l is not None:
                return l
        except (tf.errors.NotFoundError, OSError) as e:
            errs.append(str(e))
    raise NotImplementedError('unable to open file: ' + f'{filename}, from paths: {filenames}\ncaused by: {errs}')