import json
import logging
import math
import random
import threading
import time
import uuid
from functools import reduce
from typing import List, Optional
import sqlalchemy
import sqlalchemy.sql.expression as sql
from sqlalchemy import and_, func, sql, text
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities import (
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.entities.metric import MetricWithRunId
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.db.db_types import MSSQL, MYSQL
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_MAX_RESULTS_THRESHOLD
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import (
from mlflow.utils.file_utils import local_file_uri_to_path, mkdir
from mlflow.utils.mlflow_tags import (
from mlflow.utils.name_utils import _generate_random_name
from mlflow.utils.search_utils import SearchExperimentsUtils, SearchUtils
from mlflow.utils.string_utils import is_string_type
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import (
from mlflow.utils.validation import (
def _try_insert_tags(attempt_number, max_retries):
    try:
        current_tags = session.query(SqlTag).filter(SqlTag.run_uuid == run_id, SqlTag.key.in_([t.key for t in tags])).all()
        current_tags = {t.key: t for t in current_tags}
        new_tag_dict = {}
        for tag in tags:
            if tag.key == MLFLOW_RUN_NAME:
                self.set_tag(run_id, tag)
            else:
                current_tag = current_tags.get(tag.key)
                new_tag = new_tag_dict.get(tag.key)
                if current_tag:
                    current_tag.value = tag.value
                    continue
                if new_tag:
                    new_tag.value = tag.value
                else:
                    new_tag = SqlTag(run_uuid=run_id, key=tag.key, value=tag.value)
                new_tag_dict[tag.key] = new_tag
        session.add_all(new_tag_dict.values())
        session.commit()
    except sqlalchemy.exc.IntegrityError:
        session.rollback()
        if attempt_number > max_retries:
            raise MlflowException('Failed to set tags with given within {} retries. Keys: {}'.format(max_retries, [t.key for t in tags]))
        sleep_duration = 2 ** attempt_number - 1
        sleep_duration += random.uniform(0, 1)
        time.sleep(sleep_duration)
        _try_insert_tags(attempt_number + 1, max_retries=max_retries)