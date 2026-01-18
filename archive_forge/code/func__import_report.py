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
def _import_report(self, report: Report, *, namespace: Optional[Namespace]=None) -> None:
    """Import one wandb.Report.

        Use `namespace` to specify alternate settings like where the report should be uploaded
        """
    if namespace is None:
        namespace = Namespace(report.entity, report.project)
    entity = coalesce(namespace.entity, report.entity)
    project = coalesce(namespace.project, report.project)
    name = report.name
    title = report.title
    description = report.description
    api = self.dst_api
    logger.debug(f'Upserting entity={entity!r}/project={project!r}')
    try:
        api.create_project(project, entity)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code != 409:
            logger.warn(f'Issue upserting entity={entity!r}/project={project!r}, e={e!r}')
    logger.debug(f'Upserting report entity={entity!r}, project={project!r}, name={name!r}, title={title!r}')
    api.client.execute(wr.report.UPSERT_VIEW, variable_values={'id': None, 'name': name, 'entityName': entity, 'projectName': project, 'description': description, 'displayName': title, 'type': 'runs', 'spec': json.dumps(report.spec)})