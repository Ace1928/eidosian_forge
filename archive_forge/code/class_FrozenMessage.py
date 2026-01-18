from .messages import Message
from .midifiles import MetaMessage, UnknownMetaMessage
class FrozenMessage(Frozen, Message):
    pass