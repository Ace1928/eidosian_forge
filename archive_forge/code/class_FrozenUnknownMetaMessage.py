from .messages import Message
from .midifiles import MetaMessage, UnknownMetaMessage
class FrozenUnknownMetaMessage(Frozen, UnknownMetaMessage):

    def __repr__(self):
        return 'Frozen' + UnknownMetaMessage.__repr__(self)