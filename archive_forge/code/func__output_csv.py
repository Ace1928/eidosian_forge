from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable
import torch
import torch.fft
from torch.utils import benchmark
from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer
def _output_csv(file, results):
    file.write('benchmark,device,num_threads,numel,shape,contiguous,dim,mean (us),median (us),iqr (us)\n')
    for measurement in results:
        metadata = measurement.metadata
        device, dim, shape, name, numel, contiguous = (metadata['device'], metadata['dim'], metadata['shape'], metadata['name'], metadata['numel'], metadata['is_contiguous'])
        if isinstance(dim, Iterable):
            dim_str = '-'.join((str(d) for d in dim))
        else:
            dim_str = str(dim)
            shape_str = 'x'.join((str(s) for s in shape))
        print(name, device, measurement.task_spec.num_threads, numel, shape_str, contiguous, dim_str, measurement.mean * 1000000.0, measurement.median * 1000000.0, measurement.iqr * 1000000.0, sep=',', file=file)