import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _ring_permutations(num_workers, num_subchunks, gpu_perm):
    """"Generate an array of device index arrays, one for each subchunk.

  In the basic ring reduction algorithm there are size(T)/num_devices
  data chunks and each device process one chunk per tick, i.e. sending
  one chunk and receiving one chunk.  The idea of subchunking is that
  each device processes num_subchunks smaller data regions per tick,
  and the ring rank permutation is different for each subchunk index
  so that a device is potentially sending to and receiving from
  num_subchunks different other devices at each tick.  Where multiple
  independent data channels exist between devices, this strategy
  supplies a method of using them in parallel.

  Args:
    num_workers: number of worker tasks
    num_subchunks: number of subchunks into which to divide each per-GPU chunk.
    gpu_perm: an array of integers in [0, num_gpus-1] giving the default
      ring order of GPUs at each worker.  Other permutations will be generated
      by rotating this array and splicing together per-worker instances.

  Raises:
    ValueError: the number of subchunks may not exceed the number of GPUs.

  Returns:
    pred_by_s_d: list of lists that maps (by index) from (subchunk, dev) to
        preceding device in the permutation for that subchunk.  The
        device index of GPU i at worker j is i + (j * num_gpus).
    rank_by_s_d: list of lists that maps (by index) from (subchunk, dev) to
       local rank of device d in the permutation for that subchunk.
  """
    num_gpus = len(gpu_perm)
    devices = num_workers * num_gpus
    if devices == 0:
        return ([], [])
    if num_subchunks > num_gpus:
        raise ValueError('num_subchunks %d must be <= num_gpus %d' % (num_subchunks, num_gpus))
    rotation_interval = max(1, int(num_gpus / num_subchunks))
    perms_by_s = []
    for s in range(0, num_subchunks):
        full_order = []
        offset = s * rotation_interval
        for w in range(0, num_workers):
            default_order = [w * num_gpus + i for i in gpu_perm]
            dev_order = default_order[offset:] + default_order[:offset]
            full_order += dev_order
        perms_by_s.append(full_order)
    pred_by_s_d = [[-1 for d in range(0, devices)] for s in range(0, num_subchunks)]
    rank_by_s_d = [[-1 for d in range(0, devices)] for s in range(0, num_subchunks)]
    for s in range(0, num_subchunks):
        for d in range(0, devices):
            for t in range(0, devices):
                if d == perms_by_s[s][t]:
                    rank_by_s_d[s][d] = t
                    pred_by_s_d[s][d] = perms_by_s[s][(t + devices - 1) % devices]
                    break
    return (pred_by_s_d, rank_by_s_d)