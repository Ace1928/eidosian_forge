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
def _possible_configs_for_cls(cls, reasons=None, sparse=False):
    all_configs = set(config.Config.all_configs())
    if cls.__unsupported_on__:
        spec = exclusions.db_spec(*cls.__unsupported_on__)
        for config_obj in list(all_configs):
            if spec(config_obj):
                all_configs.remove(config_obj)
    if getattr(cls, '__only_on__', None):
        spec = exclusions.db_spec(*util.to_list(cls.__only_on__))
        for config_obj in list(all_configs):
            if not spec(config_obj):
                all_configs.remove(config_obj)
    if getattr(cls, '__only_on_config__', None):
        all_configs.intersection_update([cls.__only_on_config__])
    if hasattr(cls, '__requires__'):
        requirements = config.requirements
        for config_obj in list(all_configs):
            for requirement in cls.__requires__:
                check = getattr(requirements, requirement)
                skip_reasons = check.matching_config_reasons(config_obj)
                if skip_reasons:
                    all_configs.remove(config_obj)
                    if reasons is not None:
                        reasons.extend(skip_reasons)
                    break
    if hasattr(cls, '__prefer_requires__'):
        non_preferred = set()
        requirements = config.requirements
        for config_obj in list(all_configs):
            for requirement in cls.__prefer_requires__:
                check = getattr(requirements, requirement)
                if not check.enabled_for_config(config_obj):
                    non_preferred.add(config_obj)
        if all_configs.difference(non_preferred):
            all_configs.difference_update(non_preferred)
    if sparse:
        per_dialect = {}
        for cfg in reversed(sorted(all_configs, key=lambda cfg: (cfg.db.name, cfg.db.driver, cfg.db.dialect.server_version_info))):
            db = cfg.db.name
            if db not in per_dialect:
                per_dialect[db] = cfg
        return per_dialect.values()
    return all_configs