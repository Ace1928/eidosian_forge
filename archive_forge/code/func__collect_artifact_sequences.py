import itertools
import json
import logging
import numbers
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from unittest.mock import patch
import filelock
import polars as pl
import requests
import urllib3
import yaml
from wandb_gql import gql
import wandb
import wandb.apis.reports as wr
from wandb.apis.public import ArtifactCollection, Run
from wandb.apis.public.files import File
from wandb.apis.reports import Report
from wandb.util import coalesce, remove_keys_with_none_values
from . import validation
from .internals import internal
from .internals.protocols import PathStr, Policy
from .internals.util import Namespace, for_each
def _collect_artifact_sequences(self, *, namespaces: Optional[Iterable[Namespace]]=None, limit: Optional[int]=None, api: Optional[Api]=None):
    api = coalesce(api, self.src_api)
    namespaces = coalesce(namespaces, self._all_namespaces())

    def artifact_sequences():
        for ns in namespaces:
            logger.debug(f'Collecting artifact sequences from ns={ns!r}')
            types = []
            try:
                types = [t for t in api.artifact_types(ns.path)]
            except Exception as e:
                logger.error(f'Failed to get artifact types e={e!r}')
            for t in types:
                collections = []
                if t.name == 'wandb-history':
                    continue
                try:
                    collections = t.collections()
                except Exception as e:
                    logger.error(f'Failed to get artifact collections e={e!r}')
                for c in collections:
                    if c.is_sequence():
                        yield ArtifactSequence.from_collection(c)
    seqs = itertools.islice(artifact_sequences(), limit)
    unique_sequences = {seq.identifier: seq for seq in seqs}
    yield from unique_sequences.values()