from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class _AggWritePref:
    """Agg $out/$merge write preference.

    * If there are readable servers and there is any pre-5.0 server, use
      primary read preference.
    * Otherwise use `pref` read preference.

    :Parameters:
      - `pref`: The read preference to use on MongoDB 5.0+.
    """
    __slots__ = ('pref', 'effective_pref')

    def __init__(self, pref: _ServerMode):
        self.pref = pref
        self.effective_pref: _ServerMode = ReadPreference.PRIMARY

    def selection_hook(self, topology_description: TopologyDescription) -> None:
        common_wv = topology_description.common_wire_version
        if topology_description.has_readable_server(ReadPreference.PRIMARY_PREFERRED) and common_wv and (common_wv < 13):
            self.effective_pref = ReadPreference.PRIMARY
        else:
            self.effective_pref = self.pref

    def __call__(self, selection: Selection) -> Selection:
        """Apply this read preference to a Selection."""
        return self.effective_pref(selection)

    def __repr__(self) -> str:
        return f'_AggWritePref(pref={self.pref!r})'

    def __getattr__(self, name: str) -> Any:
        return getattr(self.effective_pref, name)