import typing
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.remote.webelement import WebElement
from .input_device import InputDevice
from .interaction import POINTER
from .interaction import POINTER_KINDS
def _convert_keys(self, actions: typing.Dict[str, typing.Any]):
    out = {}
    for k, v in actions.items():
        if v is None:
            continue
        if k in ('x', 'y'):
            out[k] = int(v)
            continue
        splits = k.split('_')
        new_key = splits[0] + ''.join((v.title() for v in splits[1:]))
        out[new_key] = v
    return out