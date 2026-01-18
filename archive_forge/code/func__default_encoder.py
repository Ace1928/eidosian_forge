import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _default_encoder(sample: Dict[str, Any], format: Optional[Union[str, bool]]=True):
    """A default encoder for webdataset.

    This handles common file extensions: .txt, .cls, .cls2, .jpg,
        .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl
    These are the most common extensions used in webdataset.
    For other extensions, users can provide their own encoder.

    Args:
        sample (Dict[str, Any]): sample
    """
    sample = dict(sample)
    for key, value in sample.items():
        extension = key.split('.')[-1]
        if key.startswith('__'):
            continue
        elif extension in ['txt']:
            sample[key] = value.encode('utf-8')
        elif extension in ['cls', 'cls2']:
            sample[key] = str(value).encode('utf-8')
        elif extension in ['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm']:
            import numpy as np
            import PIL.Image
            if isinstance(value, np.ndarray):
                value = PIL.Image.fromarray(value)
            assert isinstance(value, PIL.Image.Image)
            stream = io.BytesIO()
            value.save(stream, format=extension_to_format.get(extension.lower(), extension))
            sample[key] = stream.getvalue()
        elif extension == 'json':
            import json
            sample[key] = json.dumps(value).encode('utf-8')
        elif extension == 'npy':
            import numpy as np
            stream = io.BytesIO()
            np.save(stream, value)
            sample[key] = stream.getvalue()
        elif extension == 'mp':
            import msgpack
            sample[key] = msgpack.dumps(value)
        elif extension in ['pt', 'pth']:
            import torch
            stream = io.BytesIO()
            torch.save(value, stream)
            sample[key] = stream.getvalue()
        elif extension in ['pickle', 'pkl']:
            import pickle
            stream = io.BytesIO()
            pickle.dump(value, stream)
            sample[key] = stream.getvalue()
    return sample