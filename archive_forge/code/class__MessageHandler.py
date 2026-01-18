import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class _MessageHandler:
    """Message Handler class for GStreamer Sources.
    
    This separate class holds a weak reference to the
    Source, preventing garbage collection issues. 
    
    """

    def __init__(self, source):
        self.source = weakref.proxy(source)

    def message(self, bus, message):
        """The main message callback"""
        if message.type == Gst.MessageType.EOS:
            self.source.queue.put(self.source.sentinal)
            if not self.source.caps:
                raise GStreamerDecodeException('Appears to be an unsupported file')
        elif message.type == Gst.MessageType.ERROR:
            raise GStreamerDecodeException(message.parse_error())

    def notify_caps(self, pad, *args):
        """notify::caps callback"""
        self.source.caps = True
        info = pad.get_current_caps().get_structure(0)
        self.source._duration = pad.get_peer().query_duration(Gst.Format.TIME).duration / Gst.SECOND
        channels = info.get_int('channels')[1]
        sample_rate = info.get_int('rate')[1]
        sample_size = int(''.join(filter(str.isdigit, info.get_string('format'))))
        self.source.audio_format = AudioFormat(channels=channels, sample_size=sample_size, sample_rate=sample_rate)
        self.source.is_ready.set()

    def pad_added(self, element, pad):
        """pad-added callback"""
        name = pad.query_caps(None).to_string()
        if name.startswith('audio/x-raw'):
            nextpad = self.source.converter.get_static_pad('sink')
            if not nextpad.is_linked():
                self.source.pads = True
                pad.link(nextpad)

    def no_more_pads(self, element):
        """Finished Adding pads"""
        if not self.source.pads:
            raise GStreamerDecodeException('No Streams Found')

    def new_sample(self, sink):
        """new-sample callback"""
        buffer = sink.emit('pull-sample').get_buffer()
        mem = buffer.extract_dup(0, buffer.get_size())
        self.source.queue.put(mem)
        return Gst.FlowReturn.OK

    @staticmethod
    def unknown_type(uridecodebin, decodebin, caps):
        """unknown-type callback for unreadable files"""
        streaminfo = caps.to_string()
        if not streaminfo.startswith('audio/'):
            return
        raise GStreamerDecodeException(streaminfo)