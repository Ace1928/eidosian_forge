import collections
from typing import Dict
from .psparser import LIT
class PDFColorSpace:

    def __init__(self, name: str, ncomponents: int) -> None:
        self.name = name
        self.ncomponents = ncomponents

    def __repr__(self) -> str:
        return '<PDFColorSpace: %s, ncomponents=%d>' % (self.name, self.ncomponents)