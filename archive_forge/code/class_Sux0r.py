from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class Sux0r(sux.XMLParser):

    def __init__(self) -> None:
        self.tokens: list[tuple[Literal['start'], str, dict[str, str]] | tuple[Literal['text'], str]] = []

    def getTagStarts(self) -> list[tuple[Literal['start'], str, dict[str, str]]]:
        return [token for token in self.tokens if token[0] == 'start']

    def gotTagStart(self, name: str, attrs: dict[str, str]) -> None:
        self.tokens.append(('start', name, attrs))

    def gotText(self, text: str) -> None:
        self.tokens.append(('text', text))