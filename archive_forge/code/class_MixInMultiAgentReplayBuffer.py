import collections
import platform
import random
from typing import Optional
from ray.util.timer import _Timer
from ray.rllib.execution.replay_ops import SimpleReplayBuffer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.typing import PolicyID, SampleBatchType
class MixInMultiAgentReplayBuffer:
    """This buffer adds replayed samples to a stream of new experiences.

    - Any newly added batch (`add()`) is immediately returned upon
    the next `replay` call (close to on-policy) as well as being moved
    into the buffer.
    - Additionally, a certain number of old samples is mixed into the
    returned sample according to a given "replay ratio".
    - If >1 calls to `add()` are made without any `replay()` calls
    in between, all newly added batches are returned (plus some older samples
    according to the "replay ratio").

    .. testcode::

        from ray.rllib.execution.buffers.mixin_replay_buffer import (
            MixInMultiAgentReplayBuffer)
        from ray.rllib.policy.sample_batch import SampleBatch
        # replay ratio 0.66 (2/3 replayed, 1/3 new samples):
        buffer = MixInMultiAgentReplayBuffer(capacity=100,
                                             replay_ratio=0.66)
        A, B, C = (SampleBatch({"obs": [1]}), SampleBatch({"obs": [2]}),
            SampleBatch({"obs": [3]}))
        buffer.add(A)
        buffer.add(B)
        buffer.add(B)
        print(buffer.replay()["obs"])

    .. testoutput::
        :hide:

        ...
    """

    def __init__(self, capacity: int, replay_ratio: float, replay_mode: ReplayMode=ReplayMode.INDEPENDENT):
        """Initializes MixInReplay instance.

        Args:
            capacity: Number of batches to store in total.
            replay_ratio: Ratio of replayed samples in the returned
                batches. E.g. a ratio of 0.0 means only return new samples
                (no replay), a ratio of 0.5 means always return newest sample
                plus one old one (1:1), a ratio of 0.66 means always return
                the newest sample plus 2 old (replayed) ones (1:2), etc...
        """
        self.capacity = capacity
        self.replay_ratio = replay_ratio
        self.replay_proportion = None
        if self.replay_ratio != 1.0:
            self.replay_proportion = self.replay_ratio / (1.0 - self.replay_ratio)
        if replay_mode in ['lockstep', ReplayMode.LOCKSTEP]:
            self.replay_mode = ReplayMode.LOCKSTEP
        elif replay_mode in ['independent', ReplayMode.INDEPENDENT]:
            self.replay_mode = ReplayMode.INDEPENDENT
        else:
            raise ValueError('Unsupported replay mode: {}'.format(replay_mode))

        def new_buffer():
            return SimpleReplayBuffer(num_slots=capacity)
        self.replay_buffers = collections.defaultdict(new_buffer)
        self.add_batch_timer = _Timer()
        self.replay_timer = _Timer()
        self.update_priorities_timer = _Timer()
        self.num_added = 0
        self.last_added_batches = collections.defaultdict(list)

    def add(self, batch: SampleBatchType) -> None:
        """Adds a batch to the appropriate policy's replay buffer.

        Turns the batch into a MultiAgentBatch of the DEFAULT_POLICY_ID if
        it is not a MultiAgentBatch. Subsequently adds the individual policy
        batches to the storage.

        Args:
            batch: The batch to be added.
        """
        batch = batch.copy()
        batch = batch.as_multi_agent()
        with self.add_batch_timer:
            if self.replay_mode == ReplayMode.LOCKSTEP:
                self.replay_buffers[_ALL_POLICIES].add_batch(batch)
                self.last_added_batches[_ALL_POLICIES].append(batch)
            else:
                for policy_id, sample_batch in batch.policy_batches.items():
                    self.replay_buffers[policy_id].add_batch(sample_batch)
                    self.last_added_batches[policy_id].append(sample_batch)
        self.num_added += batch.count

    def replay(self, policy_id: PolicyID=DEFAULT_POLICY_ID) -> Optional[SampleBatchType]:
        if self.replay_mode == ReplayMode.LOCKSTEP and policy_id != _ALL_POLICIES:
            raise ValueError("Trying to sample from single policy's buffer in lockstep mode. In lockstep mode, all policies' experiences are sampled from a single replay buffer which is accessed with the policy id `{}`".format(_ALL_POLICIES))
        buffer = self.replay_buffers[policy_id]
        if len(buffer) == 0 or (len(self.last_added_batches[policy_id]) == 0 and self.replay_ratio < 1.0):
            return None
        with self.replay_timer:
            output_batches = self.last_added_batches[policy_id]
            self.last_added_batches[policy_id] = []
            if self.replay_ratio == 0.0:
                return concat_samples(output_batches)
            elif self.replay_ratio == 1.0:
                return buffer.replay()
            num_new = len(output_batches)
            replay_proportion = self.replay_proportion
            while random.random() < num_new * replay_proportion:
                replay_proportion -= 1
                output_batches.append(buffer.replay())
            return concat_samples(output_batches)

    def get_host(self) -> str:
        """Returns the computer's network name.

        Returns:
            The computer's networks name or an empty string, if the network
            name could not be determined.
        """
        return platform.node()

    @Deprecated(new='MixInMultiAgentReplayBuffer.add()', error=True)
    def add_batch(self, *args, **kwargs):
        pass