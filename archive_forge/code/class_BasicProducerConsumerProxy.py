from zope.interface import implementer
from twisted.internet import interfaces
@implementer(interfaces.IProducer, interfaces.IConsumer)
class BasicProducerConsumerProxy:
    """
    I can act as a man in the middle between any Producer and Consumer.

    @ivar producer: the Producer I subscribe to.
    @type producer: L{IProducer<interfaces.IProducer>}
    @ivar consumer: the Consumer I publish to.
    @type consumer: L{IConsumer<interfaces.IConsumer>}
    @ivar paused: As a Producer, am I paused?
    @type paused: bool
    """
    consumer = None
    producer = None
    producerIsStreaming = None
    iAmStreaming = True
    outstandingPull = False
    paused = False
    stopped = False

    def __init__(self, consumer):
        self._buffer = []
        if consumer is not None:
            self.consumer = consumer
            consumer.registerProducer(self, self.iAmStreaming)

    def pauseProducing(self):
        self.paused = True
        if self.producer:
            self.producer.pauseProducing()

    def resumeProducing(self):
        self.paused = False
        if self._buffer:
            self.consumer.write(''.join(self._buffer))
            self._buffer[:] = []
        elif not self.iAmStreaming:
            self.outstandingPull = True
        if self.producer is not None:
            self.producer.resumeProducing()

    def stopProducing(self):
        if self.producer is not None:
            self.producer.stopProducing()
        if self.consumer is not None:
            del self.consumer

    def write(self, data):
        if self.paused or (not self.iAmStreaming and (not self.outstandingPull)):
            self._buffer.append(data)
        elif self.consumer is not None:
            self.consumer.write(data)
            self.outstandingPull = False

    def finish(self):
        if self.consumer is not None:
            self.consumer.finish()
        self.unregisterProducer()

    def registerProducer(self, producer, streaming):
        self.producer = producer
        self.producerIsStreaming = streaming

    def unregisterProducer(self):
        if self.producer is not None:
            del self.producer
            del self.producerIsStreaming
        if self.consumer:
            self.consumer.unregisterProducer()

    def __repr__(self) -> str:
        return f'<{self.__class__}@{id(self):x} around {self.consumer}>'