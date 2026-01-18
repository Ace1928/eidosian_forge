from functools import lru_cache
from typing import Callable, Iterable, Iterator, TypeVar, Union, overload
import setuptools.extern.jaraco.text as text
from setuptools.extern.packaging.requirements import Requirement

    Replacement for ``pkg_resources.parse_requirements`` that uses ``packaging``.
    