from __future__ import annotations
import importlib.metadata
from typing import Tuple, Type
from simpy.core import Environment
from simpy.events import AllOf, AnyOf, Event, Process, Timeout
from simpy.exceptions import Interrupt, SimPyException
from simpy.resources.container import Container
from simpy.resources.resource import PreemptiveResource, PriorityResource, Resource
from simpy.resources.store import FilterStore, PriorityItem, PriorityStore, Store
from simpy.rt import RealtimeEnvironment
def _compile_toc(entries: Tuple[Tuple[str, Tuple[Type, ...]], ...], section_marker: str='=') -> str:
    """Compiles a list of sections with objects into sphinx formatted
    autosummary directives."""
    toc = ''
    for section, objs in entries:
        toc += '\n\n'
        toc += f'{section}\n'
        toc += f'{section_marker * len(section)}\n\n'
        toc += '.. autosummary::\n\n'
        for obj in objs:
            toc += f'    ~{obj.__module__}.{obj.__name__}\n'
    return toc