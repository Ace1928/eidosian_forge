import threading
import rtmidi
from .. import ports
from ..messages import Message
from ._parser_queue import ParserQueue
from .rtmidi_utils import expand_alsa_port_name
def _get_api_lookup():
    api_to_name = {}
    name_to_api = {}
    for name in dir(rtmidi):
        if name.startswith('API_'):
            value = getattr(rtmidi, name)
            name = name.replace('API_', '')
            name_to_api[name] = value
            api_to_name[value] = name
    return (api_to_name, name_to_api)