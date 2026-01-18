import collections
import dataclasses
import threading
from typing import Optional
from tensorboard.backend.event_processing import directory_loader
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import event_util
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.backend.event_processing import plugin_asset_util
from tensorboard.backend.event_processing import reservoir
from tensorboard.backend.event_processing import tag_types
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.compat.proto import meta_graph_pb2
from tensorboard.compat.proto import tensor_pb2
from tensorboard.util import tb_logging
def _Purge(self, event, by_tags):
    """Purge all events that have occurred after the given event.step.

        If by_tags is True, purge all events that occurred after the given
        event.step, but only for the tags that the event has. Non-sequential
        event.steps suggest that a TensorFlow restart occurred, and we discard
        the out-of-order events to display a consistent view in TensorBoard.

        Discarding by tags is the safer method, when we are unsure whether a restart
        has occurred, given that threading in supervisor can cause events of
        different tags to arrive with unsynchronized step values.

        If by_tags is False, then purge all events with event.step greater than the
        given event.step. This can be used when we are certain that a TensorFlow
        restart has occurred and these events can be discarded.

        Args:
          event: The event to use as reference for the purge. All events with
            the same tags, but with a greater event.step will be purged.
          by_tags: Bool to dictate whether to discard all out-of-order events or
            only those that are associated with the given reference event.
        """
    _NotExpired = lambda x: x.step < event.step
    num_expired = 0
    if by_tags:
        for value in event.summary.value:
            if value.tag in self.tensors_by_tag:
                tag_reservoir = self.tensors_by_tag[value.tag]
                num_expired += tag_reservoir.FilterItems(_NotExpired, _TENSOR_RESERVOIR_KEY)
    else:
        for tag_reservoir in self.tensors_by_tag.values():
            num_expired += tag_reservoir.FilterItems(_NotExpired, _TENSOR_RESERVOIR_KEY)
    if num_expired > 0:
        purge_msg = _GetPurgeMessage(self.most_recent_step, self.most_recent_wall_time, event.step, event.wall_time, num_expired)
        logger.warning(purge_msg)