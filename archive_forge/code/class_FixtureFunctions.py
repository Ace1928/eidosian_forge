from __future__ import annotations
import abc
from argparse import Namespace
import configparser
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any
from sqlalchemy.testing import asyncio
class FixtureFunctions(abc.ABC):

    @abc.abstractmethod
    def skip_test_exception(self, *arg, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def combinations(self, *args, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def param_ident(self, *args, **kw):
        raise NotImplementedError()

    @abc.abstractmethod
    def fixture(self, *arg, **kw):
        raise NotImplementedError()

    def get_current_test_name(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mark_base_test_class(self) -> Any:
        raise NotImplementedError()

    @abc.abstractproperty
    def add_to_marker(self):
        raise NotImplementedError()