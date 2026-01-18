from pygame import midi
from ..ports import BaseInput, BaseOutput
def _get_default_device(get_input):
    if get_input:
        device_id = midi.get_default_input_id()
    else:
        device_id = midi.get_default_output_id()
    if device_id < 0:
        raise OSError('no default port found')
    return _get_device(device_id)