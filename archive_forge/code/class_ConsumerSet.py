from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from . import messaging
from .entity import Exchange, Queue
class ConsumerSet(messaging.Consumer):

    def __init__(self, connection, from_dict=None, consumers=None, channel=None, **kwargs):
        if channel:
            self._provided_channel = True
            self.backend = channel
        else:
            self._provided_channel = False
            self.backend = connection.channel()
        queues = []
        if consumers:
            for consumer in consumers:
                queues.extend(consumer.queues)
        if from_dict:
            for queue_name, queue_options in from_dict.items():
                queues.append(Queue.from_dict(queue_name, **queue_options))
        super().__init__(self.backend, queues, **kwargs)

    def iterconsume(self, limit=None, no_ack=False):
        return _iterconsume(self.connection, self, no_ack, limit)

    def discard_all(self):
        return self.purge()

    def add_consumer_from_dict(self, queue, **options):
        return self.add_queue(Queue.from_dict(queue, **options))

    def add_consumer(self, consumer):
        for queue in consumer.queues:
            self.add_queue(queue)

    def revive(self, channel):
        self.backend = channel
        super().revive(channel)

    def close(self):
        self.cancel()
        if not self._provided_channel:
            self.channel.close()