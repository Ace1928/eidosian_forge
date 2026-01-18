import threading
import rtmidi
from .. import ports
from ..messages import Message
from ._parser_queue import ParserQueue
from .rtmidi_utils import expand_alsa_port_name
def _open_port(rt, name=None, client_name=None, virtual=False, api=None):
    if client_name is not None:
        virtual = True
    if virtual:
        if name is None:
            raise OSError('virtual port must have a name')
        rt.open_virtual_port(name)
        return name
    if api == 'LINUX_ALSA':
        name = expand_alsa_port_name(rt.get_ports(), name)
    port_names = rt.get_ports()
    if len(port_names) == 0:
        raise OSError('no ports available')
    if name is None:
        name = port_names[0]
        port_id = 0
    elif name in port_names:
        port_id = port_names.index(name)
    else:
        raise OSError(f'unknown port {name!r}')
    try:
        rt.open_port(port_id)
    except RuntimeError as err:
        raise OSError(*err.args) from err
    return name