import ipaddress
from abc import ABCMeta
from typing import Dict, List, Optional, Type, Union
from geoip2.mixins import SimpleEquality
class PlaceRecord(Record, metaclass=ABCMeta):
    """All records with :py:attr:`names` subclass :py:class:`PlaceRecord`."""
    names: Dict[str, str]
    _locales: List[str]

    def __init__(self, locales: Optional[List[str]]=None, names: Optional[Dict[str, str]]=None) -> None:
        if locales is None:
            locales = ['en']
        self._locales = locales
        if names is None:
            names = {}
        self.names = names

    @property
    def name(self) -> Optional[str]:
        """Dict with locale codes as keys and localized name as value."""
        return next((self.names.get(x) for x in self._locales if x in self.names), None)