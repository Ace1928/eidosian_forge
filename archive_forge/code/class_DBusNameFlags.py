from .low_level import Message, MessageType, HeaderFields
from .wrappers import MessageGenerator, new_method_call
class DBusNameFlags:
    allow_replacement = 1
    replace_existing = 2
    do_not_queue = 4