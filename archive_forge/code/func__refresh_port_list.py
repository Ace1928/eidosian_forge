import ctypes
import threading
from ..ports import BaseInput, BaseOutput, sleep
from . import portmidi_init as pm
def _refresh_port_list():
    if _state['port_count'] == 0:
        pm.lib.Pm_Terminate()
        pm.lib.Pm_Initialize()