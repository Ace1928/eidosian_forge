from importlib.metadata import EntryPoint, entry_points
from typing import (
import rdflib.plugins.stores.berkeleydb
from rdflib.exceptions import Error
from rdflib.parser import Parser
from rdflib.query import (
from rdflib.serializer import Serializer
from rdflib.store import Store
class PKGPlugin(Plugin[PluginT]):

    def __init__(self, name: str, kind: Type[PluginT], ep: 'EntryPoint'):
        self.name = name
        self.kind = kind
        self.ep = ep
        self._class: Optional[Type[PluginT]] = None

    def getClass(self) -> Type[PluginT]:
        if self._class is None:
            self._class = self.ep.load()
        return self._class