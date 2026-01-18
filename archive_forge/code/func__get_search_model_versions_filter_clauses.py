import logging
import urllib
import sqlalchemy
from sqlalchemy.future import select
import mlflow.store.db.utils
from mlflow.entities.model_registry.model_version_stages import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.utils.models import _parse_model_uri
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry import (
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.store.model_registry.dbmodels.models import (
from mlflow.utils.search_utils import SearchModelUtils, SearchModelVersionUtils, SearchUtils
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.uri import extract_db_type_from_uri
from mlflow.utils.validation import (
@classmethod
def _get_search_model_versions_filter_clauses(cls, parsed_filters, dialect):
    attribute_filters = []
    tag_filters = {}
    for f in parsed_filters:
        type_ = f['type']
        key = f['key']
        comparator = f['comparator']
        value = f['value']
        if type_ == 'attribute':
            if key not in SearchModelVersionUtils.VALID_SEARCH_ATTRIBUTE_KEYS:
                raise MlflowException(f'Invalid attribute name: {key}', error_code=INVALID_PARAMETER_VALUE)
            if key in SearchModelVersionUtils.NUMERIC_ATTRIBUTES:
                if comparator not in SearchModelVersionUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS:
                    raise MlflowException(f'Invalid comparator for attribute {key}: {comparator}', error_code=INVALID_PARAMETER_VALUE)
            elif comparator not in SearchModelVersionUtils.VALID_STRING_ATTRIBUTE_COMPARATORS or (comparator == 'IN' and key != 'run_id'):
                raise MlflowException(f'Invalid comparator for attribute: {comparator}', error_code=INVALID_PARAMETER_VALUE)
            if key == 'source_path':
                key_name = 'source'
            elif key == 'version_number':
                key_name = 'version'
            else:
                key_name = key
            attr = getattr(SqlModelVersion, key_name)
            if comparator == 'IN':
                val_filter = attr.in_(value)
            else:
                val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(attr, value)
            attribute_filters.append(val_filter)
        elif type_ == 'tag':
            if comparator not in ('=', '!=', 'LIKE', 'ILIKE'):
                raise MlflowException.invalid_parameter_value(f'Invalid comparator for tag: {comparator}')
            if key not in tag_filters:
                key_filter = SearchUtils.get_sql_comparison_func('=', dialect)(SqlModelVersionTag.key, key)
                tag_filters[key] = [key_filter]
            val_filter = SearchUtils.get_sql_comparison_func(comparator, dialect)(SqlModelVersionTag.value, value)
            tag_filters[key].append(val_filter)
        else:
            raise MlflowException(f'Invalid token type: {type_}', error_code=INVALID_PARAMETER_VALUE)
    mv_query = select(SqlModelVersion).filter(*attribute_filters)
    if tag_filters:
        sql_tag_filters = (sqlalchemy.and_(*x) for x in tag_filters.values())
        tag_filter_query = select(SqlModelVersionTag.name, SqlModelVersionTag.version).filter(sqlalchemy.or_(*sql_tag_filters)).group_by(SqlModelVersionTag.name, SqlModelVersionTag.version).having(sqlalchemy.func.count(sqlalchemy.literal(1)) == len(tag_filters)).subquery()
        return mv_query.join(tag_filter_query, sqlalchemy.and_(SqlModelVersion.name == tag_filter_query.c.name, SqlModelVersion.version == tag_filter_query.c.version))
    else:
        return mv_query