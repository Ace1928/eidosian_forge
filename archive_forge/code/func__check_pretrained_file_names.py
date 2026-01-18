import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
@classmethod
def _check_pretrained_file_names(cls, pretrained_file_name):
    """Checks if a pre-trained token embedding file name is valid.


        Parameters
        ----------
        pretrained_file_name : str
            The pre-trained token embedding file.
        """
    embedding_name = cls.__name__.lower()
    if pretrained_file_name not in cls.pretrained_file_name_sha1:
        raise KeyError('Cannot find pretrained file %s for token embedding %s. Valid pretrained files for embedding %s: %s' % (pretrained_file_name, embedding_name, embedding_name, ', '.join(cls.pretrained_file_name_sha1.keys())))