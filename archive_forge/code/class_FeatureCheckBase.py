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
class FeatureCheckBase(metaclass=abc.ABCMeta):
    """Base class for feature version checks"""
    feature_registry: T.ClassVar[T.Dict[str, T.Dict[str, T.Set[T.Tuple[str, T.Optional['mparser.BaseNode']]]]]]
    emit_notice = False
    unconditional = False

    def __init__(self, feature_name: str, feature_version: str, extra_message: str=''):
        self.feature_name = feature_name
        self.feature_version = feature_version
        self.extra_message = extra_message

    @staticmethod
    def get_target_version(subproject: str) -> str:
        if subproject not in mesonlib.project_meson_versions:
            return ''
        return mesonlib.project_meson_versions[subproject]

    @staticmethod
    @abc.abstractmethod
    def check_version(target_version: str, feature_version: str) -> bool:
        pass

    def use(self, subproject: 'SubProject', location: T.Optional['mparser.BaseNode']=None) -> None:
        tv = self.get_target_version(subproject)
        if tv == '' and (not self.unconditional):
            return
        if self.check_version(tv, self.feature_version) and (not self.emit_notice):
            return
        if subproject not in self.feature_registry:
            self.feature_registry[subproject] = {self.feature_version: set()}
        register = self.feature_registry[subproject]
        if self.feature_version not in register:
            register[self.feature_version] = set()
        feature_key = (self.feature_name, location)
        if feature_key in register[self.feature_version]:
            return
        register[self.feature_version].add(feature_key)
        if self.check_version(tv, self.feature_version):
            return
        self.log_usage_warning(tv, location)

    @classmethod
    def report(cls, subproject: str) -> None:
        if subproject not in cls.feature_registry:
            return
        warning_str = cls.get_warning_str_prefix(cls.get_target_version(subproject))
        notice_str = cls.get_notice_str_prefix(cls.get_target_version(subproject))
        fv = cls.feature_registry[subproject]
        tv = cls.get_target_version(subproject)
        for version in sorted(fv.keys()):
            message = ', '.join(sorted({f"'{i[0]}'" for i in fv[version]}))
            if cls.check_version(tv, version):
                notice_str += '\n * {}: {{{}}}'.format(version, message)
            else:
                warning_str += '\n * {}: {{{}}}'.format(version, message)
        if '\n' in notice_str:
            mlog.notice(notice_str, fatal=False)
        if '\n' in warning_str:
            mlog.warning(warning_str)

    def log_usage_warning(self, tv: str, location: T.Optional['mparser.BaseNode']) -> None:
        raise InterpreterException('log_usage_warning not implemented')

    @staticmethod
    def get_warning_str_prefix(tv: str) -> str:
        raise InterpreterException('get_warning_str_prefix not implemented')

    @staticmethod
    def get_notice_str_prefix(tv: str) -> str:
        raise InterpreterException('get_notice_str_prefix not implemented')

    def __call__(self, f: TV_func) -> TV_func:

        @wraps(f)
        def wrapped(*wrapped_args: T.Any, **wrapped_kwargs: T.Any) -> T.Any:
            node, _, _, subproject = get_callee_args(wrapped_args)
            if subproject is None:
                raise AssertionError(f'{wrapped_args!r}')
            self.use(subproject, node)
            return f(*wrapped_args, **wrapped_kwargs)
        return T.cast('TV_func', wrapped)

    @classmethod
    def single_use(cls, feature_name: str, version: str, subproject: 'SubProject', extra_message: str='', location: T.Optional['mparser.BaseNode']=None) -> None:
        """Oneline version that instantiates and calls use()."""
        cls(feature_name, version, extra_message).use(subproject, location)