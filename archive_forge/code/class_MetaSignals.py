from __future__ import annotations
import itertools
import typing
import warnings
import weakref
class MetaSignals(type):
    """
    register the list of signals in the class variable signals,
    including signals in superclasses.
    """

    def __init__(cls, name: str, bases: tuple[type, ...], d: dict[str, typing.Any]) -> None:
        signals = d.get('signals', [])
        for superclass in cls.__bases__:
            signals.extend(getattr(superclass, 'signals', []))
        signals = list(dict.fromkeys(signals).keys())
        d['signals'] = signals
        register_signal(cls, signals)
        super().__init__(name, bases, d)