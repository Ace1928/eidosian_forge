from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@event_class('Page.javascriptDialogOpening')
@dataclass
class JavascriptDialogOpening:
    """
    Fired when a JavaScript initiated dialog (alert, confirm, prompt, or onbeforeunload) is about to
    open.
    """
    url: str
    message: str
    type_: DialogType
    has_browser_handler: bool
    default_prompt: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> JavascriptDialogOpening:
        return cls(url=str(json['url']), message=str(json['message']), type_=DialogType.from_json(json['type']), has_browser_handler=bool(json['hasBrowserHandler']), default_prompt=str(json['defaultPrompt']) if 'defaultPrompt' in json else None)