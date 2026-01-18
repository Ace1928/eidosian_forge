from pygame import midi
from ..ports import BaseInput, BaseOutput
def _get_named_device(name, get_input):
    for device in get_devices():
        if device['name'] != name:
            continue
        if get_input:
            if device['is_output']:
                continue
        elif device['is_input']:
            continue
        if device['opened']:
            raise OSError(f'port already opened: {name!r}')
        return device
    else:
        raise OSError(f'unknown port: {name!r}')