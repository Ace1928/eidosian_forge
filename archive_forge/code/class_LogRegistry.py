import functools
import itertools
import logging
import os
import re
from dataclasses import dataclass, field
from importlib import __import__
from typing import Dict, List, Optional, Set, Union
from weakref import WeakSet
import torch._guards
import torch.distributed as dist
@dataclass
class LogRegistry:
    log_alias_to_log_qnames: Dict[str, List[str]] = field(default_factory=dict)
    artifact_log_qnames: Set[str] = field(default_factory=set)
    child_log_qnames: Set[str] = field(default_factory=set)
    artifact_names: Set[str] = field(default_factory=set)
    visible_artifacts: Set[str] = field(default_factory=set)
    artifact_descriptions: Dict[str, str] = field(default_factory=dict)
    off_by_default_artifact_names: Set[str] = field(default_factory=set)
    artifact_log_formatters: Dict[str, logging.Formatter] = field(default_factory=dict)

    def is_artifact(self, name):
        return name in self.artifact_names

    def is_log(self, alias):
        return alias in self.log_alias_to_log_qnames

    def register_log(self, alias, log_qnames: Union[str, List[str]]):
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        self.log_alias_to_log_qnames[alias] = log_qnames

    def register_artifact_name(self, name, description, visible, off_by_default, log_format):
        self.artifact_names.add(name)
        if visible:
            self.visible_artifacts.add(name)
        self.artifact_descriptions[name] = description
        if off_by_default:
            self.off_by_default_artifact_names.add(name)
        if log_format is not None:
            self.artifact_log_formatters[name] = logging.Formatter(log_format)

    def register_artifact_log(self, artifact_log_qname):
        self.artifact_log_qnames.add(artifact_log_qname)

    def register_child_log(self, log_qname):
        self.child_log_qnames.add(log_qname)

    def get_log_qnames(self) -> Set[str]:
        return {qname for qnames in self.log_alias_to_log_qnames.values() for qname in qnames}

    def get_artifact_log_qnames(self):
        return set(self.artifact_log_qnames)

    def get_child_log_qnames(self):
        return set(self.child_log_qnames)

    def is_off_by_default(self, artifact_qname):
        return artifact_qname in self.off_by_default_artifact_names