from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
class PowerSGDState:
    """
    Stores both the algorithm's hyperparameters and the internal state for all the gradients during the training.
    Particularly, ``matrix_approximation_rank`` and ``start_powerSGD_iter`` are the main hyperparameters that should be tuned by the user.
    For performance, we suggest to keep binary hyperparameters ``use_error_feedback`` and ``warm_start`` on.

    1. ``matrix_approximation_rank`` controls the size of compressed low-rank tensors, which determines the compression rate. The lower the rank, the stronger the compression.

        1.1. If ``matrix_approximation_rank`` is too low, the full model quality will need more training steps to reach or will never reach and yield loss in accuracy.

        1.2. The increase of ``matrix_approximation_rank`` can substantially increase the computation costs of the compression, and the accuracy may not be further improved beyond a certain ``matrix_approximation_rank`` threshold.

    To tune ``matrix_approximation_rank``, we suggest to start from 1 and increase by factors of 2 (like an exponential grid search, 1, 2, 4, ...), until a satisfactory accuracy is reached. Typically only a small value 1-4 is used. For some NLP tasks (as shown in Appendix D of the original paper), this value has been increased to 32.

    2. ``start_powerSGD_iter`` defers PowerSGD compression until step ``start_powerSGD_iter``, and vanilla allreduce runs prior to step ``start_powerSGD_iter``. This hybrid scheme of **vanilla allreduce + PowerSGD** can effectively improve the accuracy, even a relatively small ``matrix_approximation_rank`` is used. This is because that, the beginning of training phase is usually very sensitive to inaccurate gradients, and compressing gradients too early may make the training quickly take a suboptimal trajectory, which can result in an irrecoverable impact on the accuracy.

    To tune ``start_powerSGD_iter``, we suggest to start with 10% of total training steps, and increase it until a satisfactory accuracy is reached. If there is a warm-up stage in the training, ``start_powerSGD_iter`` typically should be no less than the number of warm-up steps.

    3. ``min_compression_rate`` is the minimum compression rate required when a layer is compressed. Due to the computation overheads incurred by the compression, a tensor is worth compressing only if there can be sufficient saving in bandwidth, where ``(num_rows + num_cols) * matrix_approximation_rank * min_compression_rate < num_rows * num_cols``. If the specified compression rate threshold cannot be satisfied, the tensor will be directly allreduced without compression.

    Compression statistics are logged every ``compression_stats_logging_frequency`` iterations once PowerSGD compression starts.

    4. ``orthogonalization_epsilon`` can be a very small value (e.g., 1e-8) added to every normalized matrix column in orthogonalization step, to prevent div-by-zero error if any column has all 0s. If this can already be prevented (e.g., by batch normalization), an epsilon of 0 is recommended for accuracy.

    5. ``batch_tensors_with_same_shape`` controls whether to compress and decompress tensors with same shape in a batched operation to achieve higher parallelism. Note that you should also increase the bucket size (i.e., ``bucket_cap_mb`` arg in DDP constructor) to make more same-shaped tensors appear in the same bucket, however this may reduce the overlap between computation and communication, and increase the memory footprint due to stacking the tensors of the same shape. Set to ``True`` if the compression / decompression computation is a bottleneck.

    .. warning ::
        If error feedback or warm-up is enabled, the minimum value of ``start_powerSGD_iter`` allowed in DDP is 2.
        This is because there is another internal optimization that rebuilds buckets at iteration 1 in DDP,
        and this can conflict with any tensor memorized before the rebuild process.
    """
    __slots__ = ['process_group', 'matrix_approximation_rank', 'start_powerSGD_iter', 'min_compression_rate', 'orthogonalization_epsilon', 'use_error_feedback', 'warm_start', 'batch_tensors_with_same_shape', 'rng', 'error_dict', 'p_memory_dict', 'q_memory_dict', 'iter', 'total_numel_before_compression', 'total_numel_after_compression', 'compression_stats_logging_frequency', 'next_stats_report']

    def __init__(self, process_group, matrix_approximation_rank=1, start_powerSGD_iter=1000, min_compression_rate=2, use_error_feedback=True, warm_start=True, orthogonalization_epsilon=0, random_seed=0, compression_stats_logging_frequency=10000, batch_tensors_with_same_shape: bool=False):
        logger.info('PowerSGD config: matrix_approximation_rank = %s; start_powerSGD_iter = %s; min_compression_rate = %s; orthogonalization_epsilon = %s; use_error_feedback = %s; warm_start = %s; random_seed = %s; compression_stats_logging_frequency = %s; batch_tensors_with_same_shape = %s', matrix_approximation_rank, start_powerSGD_iter, min_compression_rate, orthogonalization_epsilon, use_error_feedback, warm_start, random_seed, compression_stats_logging_frequency, batch_tensors_with_same_shape)
        self.process_group = process_group
        self.matrix_approximation_rank = matrix_approximation_rank
        if (use_error_feedback or warm_start) and start_powerSGD_iter <= 1:
            raise ValueError('Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, because PowerSGD can only be applied after the first two iterations in DDP.')
        self.start_powerSGD_iter = start_powerSGD_iter
        self.min_compression_rate = min_compression_rate
        self.use_error_feedback = use_error_feedback
        self.warm_start = warm_start
        self.orthogonalization_epsilon = orthogonalization_epsilon
        import numpy as np
        self.rng = np.random.RandomState(random_seed)
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.p_memory_dict: Dict[int, torch.Tensor] = {}
        self.q_memory_dict: Dict[int, torch.Tensor] = {}
        self.iter = 0
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0
        self.compression_stats_logging_frequency = max(1, compression_stats_logging_frequency)
        self.next_stats_report = 0
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape

    def __getstate__(self):
        """
        Returns a ``Dict[str, Any]`` which will be pickled and saved.
        ``process_group`` is not serializable and excluded from
        a returned state.
        """
        logger.warning('NOTE: Process group is not serializable and excluded from a saved state.')
        return {slot: getattr(self, slot) for slot in self.__slots__ if slot != 'process_group'}

    def __setstate__(self, state):
        """
        Takes a provided ``state`` and retrieves ``PowerSGDState``.
        ``process_group`` is set to default.
        """
        self.process_group = distributed_c10d._get_default_group()
        logger.warning('NOTE: Process group will be set to a default group (i.e. the world size).                If a different group is desired, please set `self.process_group` after PowerSGD state is loaded.')
        for slot, value in state.items():
            setattr(self, slot, value)

    def maybe_increase_iter(self, bucket):
        if bucket.is_last():
            self.iter += 1
        if self.iter == self.start_powerSGD_iter:
            logger.info('Start to apply PowerSGD after %s iterations.', self.iter)

    def compression_stats(self):
        """
        Returns the latest compression statistics as a tuple of the form (compress_rate, numel_before_compression, numel_after_compression), where:

        compress_rate is the effective compression rate i.e. (number of elements before compression) / (number of elements after compression);

        numel_before_compression is the total number of elements before compression was applied; and,

        numel_after_compression is the total number of elements after compression was applied.
        """
        compress_rate = self.total_numel_before_compression / self.total_numel_after_compression if self.total_numel_after_compression > 0 else 0
        return (compress_rate, self.total_numel_before_compression, self.total_numel_after_compression)