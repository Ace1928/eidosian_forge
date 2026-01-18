from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class WindowControlsOverlayConfig:
    """
    Configuration for Window Controls Overlay
    """
    show_css: bool
    selected_platform: str
    theme_color: str

    def to_json(self):
        json = dict()
        json['showCSS'] = self.show_css
        json['selectedPlatform'] = self.selected_platform
        json['themeColor'] = self.theme_color
        return json

    @classmethod
    def from_json(cls, json):
        return cls(show_css=bool(json['showCSS']), selected_platform=str(json['selectedPlatform']), theme_color=str(json['themeColor']))