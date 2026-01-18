from logutils.queue import QueueHandler, QueueListener
class RedisQueueListener(QueueListener):
    """
    A QueueListener implementation which fetches pickled
    records from a Redis queue using a specified key.

    :param key: The key to use for the queue. Defaults to
                "python.logging".
    :param redis: If specified, this instance is used to
                  communicate with a Redis instance.
    """

    def __init__(self, *handlers, **kwargs):
        redis = kwargs.get('redis')
        if redis is None:
            from redis import Redis
            redis = Redis()
        self.key = kwargs.get('key', 'python.logging')
        QueueListener.__init__(self, redis, *handlers)

    def dequeue(self, block):
        """
        Dequeue and return a record.
        """
        if block:
            s = self.queue.blpop(self.key)[1]
        else:
            s = self.queue.lpop(self.key)
        if not s:
            record = None
        else:
            record = pickle.loads(s)
        return record

    def enqueue_sentinel(self):
        self.queue.rpush(self.key, '')