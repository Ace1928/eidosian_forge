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
def _import_artifact_sequence(self, seq: ArtifactSequence, *, namespace: Optional[Namespace]=None) -> None:
    """Import one artifact sequence.

        Use `namespace` to specify alternate settings like where the artifact sequence should be uploaded
        """
    if not seq.artifacts:
        logger.warn(f'Artifact seq={seq!r} has no artifacts, skipping.')
        return
    if namespace is None:
        namespace = Namespace(seq.entity, seq.project)
    settings_override = {'api_key': self.dst_api_key, 'base_url': self.dst_base_url, 'resume': 'true', 'resumed': True}
    send_manager_config = internal.SendManagerConfig(log_artifacts=True)
    self._delete_collection_in_dst(seq, namespace)
    art = seq.artifacts[0]
    run_or_dummy: Optional[Run] = _get_run_or_dummy_from_art(art, self.src_api)
    groups_of_artifacts = list(_make_groups_of_artifacts(seq))
    for i, group in enumerate(groups_of_artifacts, 1):
        art = group[0]
        if art.description == ART_SEQUENCE_DUMMY_PLACEHOLDER:
            run = WandbRun(run_or_dummy, **self.run_api_kwargs)
        else:
            try:
                wandb_run = art.logged_by()
            except ValueError:
                pass
            if wandb_run is None:
                logger.warn(f'Run for art.name={art.name!r} does not exist (deleted?), using run_or_dummy={run_or_dummy!r}')
                wandb_run = run_or_dummy
            new_art = _clone_art(art)
            group = [new_art]
            run = WandbRun(wandb_run, **self.run_api_kwargs)
        logger.info(f'Uploading partial artifact seq={seq!r}, {i}/{len(groups_of_artifacts)}')
        internal.send_run(run, extra_arts=group, overrides=namespace.send_manager_overrides, settings_override=settings_override, config=send_manager_config)
    logger.info(f'Finished uploading seq={seq!r}')
    self._remove_placeholders(seq)