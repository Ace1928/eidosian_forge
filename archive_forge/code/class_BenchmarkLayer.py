import itertools
import random
import string
import time
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src.layers.preprocessing import hashing
class BenchmarkLayer(tf.test.Benchmark):
    """Benchmark the layer forward pass."""

    def run_dataset_implementation(self, batch_size):
        num_repeats = 5
        starts = []
        ends = []
        for _ in range(num_repeats):
            ds = tf.data.Dataset.from_generator(word_gen, tf.string, tf.TensorShape([]))
            ds = ds.shuffle(batch_size * 100)
            ds = ds.batch(batch_size)
            num_batches = 5
            ds = ds.take(num_batches)
            ds = ds.prefetch(num_batches)
            starts.append(time.time())
            for i in ds:
                _ = tf.strings.to_hash_bucket(i, num_buckets=2)
            ends.append(time.time())
        avg_time = np.mean(np.array(ends) - np.array(starts)) / num_batches
        return avg_time

    def bm_layer_implementation(self, batch_size):
        input_1 = keras.Input(shape=(None,), dtype=tf.string, name='word')
        layer = hashing.Hashing(num_bins=2)
        _ = layer(input_1)
        num_repeats = 5
        starts = []
        ends = []
        for _ in range(num_repeats):
            ds = tf.data.Dataset.from_generator(word_gen, tf.string, tf.TensorShape([]))
            ds = ds.shuffle(batch_size * 100)
            ds = ds.batch(batch_size)
            num_batches = 5
            ds = ds.take(num_batches)
            ds = ds.prefetch(num_batches)
            starts.append(time.time())
            for i in ds:
                _ = layer(i)
            ends.append(time.time())
        avg_time = np.mean(np.array(ends) - np.array(starts)) / num_batches
        name = f'hashing|batch_{batch_size}'
        baseline = self.run_dataset_implementation(batch_size)
        extras = {'dataset implementation baseline': baseline, 'delta seconds': baseline - avg_time, 'delta percent': (baseline - avg_time) / baseline * 100}
        self.report_benchmark(iters=num_repeats, wall_time=avg_time, extras=extras, name=name)

    def benchmark_vocab_size_by_batch(self):
        for batch in [32, 64, 256]:
            self.bm_layer_implementation(batch_size=batch)