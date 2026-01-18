import random
from typing import (
from ...public import PanelMetricsHelper
from .validators import UNDEFINED_TYPE, TypeValidator, Validator
class InlineLaTeX(Base):
    latex: str = Attr()

    def __init__(self, latex='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latex = latex

    @property
    def spec(self) -> dict:
        return {'type': 'latex', 'children': [{'text': ''}], 'content': self.latex}