import inspect
import sys
class CommitFailedError(KafkaError):

    def __init__(self, *args, **kwargs):
        super(CommitFailedError, self).__init__('Commit cannot be completed since the group has already\n            rebalanced and assigned the partitions to another member.\n            This means that the time between subsequent calls to poll()\n            was longer than the configured max_poll_interval_ms, which\n            typically implies that the poll loop is spending too much\n            time message processing. You can address this either by\n            increasing the rebalance timeout with max_poll_interval_ms,\n            or by reducing the maximum size of batches returned in poll()\n            with max_poll_records.\n            ', *args, **kwargs)