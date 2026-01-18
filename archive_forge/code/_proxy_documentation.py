from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Iterable, cast
from typing_extensions import override
Helper method that returns the current proxy, typed as the loaded object