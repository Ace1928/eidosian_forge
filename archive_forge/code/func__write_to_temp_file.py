import os
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import index_lookup
def _write_to_temp_file(self, file_name, vocab_list):
    vocab_path = os.path.join(self.get_temp_dir(), file_name + '.txt')
    with tf.io.gfile.GFile(vocab_path, 'w') as writer:
        for vocab in vocab_list:
            writer.write(vocab + '\n')
        writer.flush()
        writer.close()
    return vocab_path