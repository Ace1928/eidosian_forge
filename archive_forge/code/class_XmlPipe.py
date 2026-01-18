from twisted.python import log
from twisted.words.xish import xpath
class XmlPipe:
    """
    XML stream pipe.

    Connects two objects that communicate stanzas through an XML stream like
    interface. Each of the ends of the pipe (sink and source) can be used to
    send XML stanzas to the other side, or add observers to process XML stanzas
    that were sent from the other side.

    XML pipes are usually used in place of regular XML streams that are
    transported over TCP. This is the reason for the use of the names source
    and sink for both ends of the pipe. The source side corresponds with the
    entity that initiated the TCP connection, whereas the sink corresponds with
    the entity that accepts that connection. In this object, though, the source
    and sink are treated equally.

    Unlike Jabber
    L{XmlStream<twisted.words.protocols.jabber.xmlstream.XmlStream>}s, the sink
    and source objects are assumed to represent an eternal connected and
    initialized XML stream. As such, events corresponding to connection,
    disconnection, initialization and stream errors are not dispatched or
    processed.

    @since: 8.2
    @ivar source: Source XML stream.
    @ivar sink: Sink XML stream.
    """

    def __init__(self):
        self.source = EventDispatcher()
        self.sink = EventDispatcher()
        self.source.send = lambda obj: self.sink.dispatch(obj)
        self.sink.send = lambda obj: self.source.dispatch(obj)