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
class LogState:
    log_qname_to_level: Dict[str, str] = field(default_factory=dict)
    artifact_names: Set[str] = field(default_factory=set)

    def enable_artifact(self, artifact_name):
        self.artifact_names.add(artifact_name)

    def is_artifact_enabled(self, name):
        return name in self.artifact_names

    def enable_log(self, log_qnames, log_level):
        if isinstance(log_qnames, str):
            log_qnames = [log_qnames]
        for log_qname in log_qnames:
            self.log_qname_to_level[log_qname] = log_level

    def get_log_level_pairs(self):
        """Returns all qualified module names for which the user requested
        explicit logging settings.

        .. warning:

            This function used to return all loggers, regardless of whether
            or not the user specified them or not; it now only returns logs
            which were explicitly mentioned by the user (and torch, which
            always is implicitly requested when we initialize our logging
            subsystem.)
        """
        return self.log_qname_to_level.items()

    def clear(self):
        self.log_qname_to_level.clear()
        self.artifact_names.clear()