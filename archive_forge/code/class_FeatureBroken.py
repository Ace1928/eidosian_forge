from __future__ import annotations
from .. import mesonlib, mlog
from .disabler import Disabler
from .exceptions import InterpreterException, InvalidArguments
from ._unholder import _unholder
from dataclasses import dataclass
from functools import wraps
import abc
import itertools
import copy
import typing as T
class FeatureBroken(FeatureCheckBase):
    """Checks for broken features"""
    feature_registry = {}
    unconditional = True

    @staticmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        return False

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        return 'Broken features used:'

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        return ''

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        args = ['Project uses feature that was always broken,', 'and is now deprecated since', f"'{self.feature_version}':", f'{self.feature_name}.']
        if self.extra_message:
            args.append(self.extra_message)
        mlog.deprecation(*args, location=location)