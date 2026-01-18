import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
class LaTeXBlock(Block):
    text: Union[str, list] = Attr()

    def __init__(self, text='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = text
        if isinstance(self.text, list):
            self.text = '\n'.join(self.text)

    @classmethod
    def from_json(cls, spec: dict) -> 'LaTeXBlock':
        text = spec['content']
        return cls(text)

    @property
    def spec(self) -> dict:
        return {'type': 'latex', 'children': [{'text': ''}], 'content': self.text, 'block': True}