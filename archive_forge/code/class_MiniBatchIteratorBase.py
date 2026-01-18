import math
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import SampleBatch
@DeveloperAPI
class MiniBatchIteratorBase:
    """The base class for all minibatch iterators.

    Args:
        batch: The input multi-agent batch.
        minibatch_size: The size of the minibatch for each module_id.
        num_iters: The number of epochs to cover. If the input batch is smaller than
            minibatch_size, then the iterator will cycle through the batch until it
            has covered num_iters epochs.
    """

    def __init__(self, batch: MultiAgentBatch, minibatch_size: int, num_iters: int=1) -> None:
        pass