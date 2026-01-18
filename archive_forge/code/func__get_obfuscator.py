from pathlib import Path
from typing import Union
import numpy as np
from collections import OrderedDict
from minerl.herobraine.hero import spaces
from minerl.herobraine.wrappers.vector_wrapper import Vectorized
from minerl.herobraine.wrapper import EnvWrapper
import copy
import os
@staticmethod
def _get_obfuscator(obfuscator_dir: Union[str, Path]):
    """Gets the obfuscator from a directory.

        Args:
            obfuscator_dir (Union[str, Path]): The directory containg the pickled obfuscators.
        """

    def make_func(np_lays):

        def func(x):
            for t, data in np_lays:
                if t == 'linear':
                    W, b = data
                    x = x.dot(W.T) + b
                elif t == 'relu':
                    x = x * (x > 0)
                elif t == 'subset_softmax':
                    discrete_subset = data
                    for a, b in discrete_subset:
                        y = x[..., a:b]
                        e_x = np.exp(y - np.max(x))
                        x[..., a:b] = e_x / e_x.sum(axis=-1)
                else:
                    raise NotImplementedError()
            return x
        return func
    assert os.path.exists(obfuscator_dir), '{} not found.'.format(obfuscator_dir)
    assert set(os.listdir(obfuscator_dir)).issuperset({OBSERVATION_OBFUSCATOR_FILE_NAME, ACTION_OBFUSCATOR_FILE_NAME, SIZE_FILE_NAME})
    with open(os.path.join(obfuscator_dir, SIZE_FILE_NAME), 'r') as f:
        obf_vector_len = int(f.read())
    ac_enc, ac_dec = np.load(os.path.join(obfuscator_dir, ACTION_OBFUSCATOR_FILE_NAME), allow_pickle=True)['arr_0']
    ac_enc, ac_dec = (make_func(ac_enc), make_func(ac_dec))
    obs_enc, obs_dec = np.load(os.path.join(obfuscator_dir, OBSERVATION_OBFUSCATOR_FILE_NAME), allow_pickle=True)['arr_0']
    obs_enc, obs_dec = (make_func(obs_enc), make_func(obs_dec))
    return (obf_vector_len, ac_enc, ac_dec, obs_enc, obs_dec)