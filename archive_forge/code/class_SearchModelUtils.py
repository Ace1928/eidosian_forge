import ast
import base64
import json
import math
import operator
import re
import shlex
import sqlparse
from packaging.version import Version
from sqlparse.sql import (
from sqlparse.tokens import Token as TokenType
from mlflow.entities import RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.utils.mlflow_tags import (
class SearchModelUtils(SearchUtils):
    NUMERIC_ATTRIBUTES = {'creation_timestamp', 'last_updated_timestamp'}
    VALID_SEARCH_ATTRIBUTE_KEYS = {'name'}
    VALID_ORDER_BY_KEYS_REGISTERED_MODELS = {'name', 'creation_timestamp', 'last_updated_timestamp'}

    @classmethod
    def _does_registered_model_match_clauses(cls, model, sed):
        key_type = sed.get('type')
        key = sed.get('key')
        value = sed.get('value')
        comparator = sed.get('comparator').upper()
        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(model, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(model, key)
            value = int(value)
        elif cls.is_tag(key_type, comparator):
            lhs = model.tags.get(key, None)
        else:
            raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
        if lhs is None:
            return False
        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, registered_models, filter_string):
        """Filters a set of registered models based on a search filter string."""
        if not filter_string:
            return registered_models
        parsed = cls.parse_search_filter(filter_string)

        def registered_model_matches(model):
            return all((cls._does_registered_model_match_clauses(model, s) for s in parsed))
        return [registered_model for registered_model in registered_models if registered_model_matches(registered_model)]

    @classmethod
    def parse_order_by_for_search_registered_models(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = SearchExperimentsUtils._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_KEYS_REGISTERED_MODELS)
        return (identifier['type'], identifier['key'], is_ascending)

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_registered_models, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == 'attribute':
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f'Invalid order_by entity: {type_}')
        if not any((key == 'name' for key, _ in order_by)):
            order_by.append(('name', True))
        return lambda model: tuple((_apply_reversor(model, k, asc) for k, asc in order_by))

    @classmethod
    def sort(cls, models, order_by_list):
        return sorted(models, key=cls._get_sort_key(order_by_list))

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_model_registry, tokens))
        if len(invalids) > 0:
            invalid_clauses = ', '.join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(f'Invalid clause(s) in filter string: {invalid_clauses}')
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _get_model_search_identifier(cls, identifier, valid_attributes):
        tokens = identifier.split('.', maxsplit=1)
        if len(tokens) == 1:
            key = tokens[0]
            identifier = cls._ATTRIBUTE_IDENTIFIER
        else:
            entity_type, key = tokens
            valid_entity_types = ('attribute', 'tag', 'tags')
            if entity_type not in valid_entity_types:
                raise MlflowException.invalid_parameter_value(f"Invalid entity type '{entity_type}'. Valid entity types are {valid_entity_types}")
            identifier = cls._TAG_IDENTIFIER if entity_type in ('tag', 'tags') else cls._ATTRIBUTE_IDENTIFIER
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'")
        key = cls._trim_backticks(cls._strip_quotes(key))
        return {'type': identifier, 'key': key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_model_search_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp['comparator'] = comparator.value.upper()
        comp['value'] = cls._get_value(comp.get('type'), comp.get('key'), right)
        return comp

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException(f"Expected a quoted string value for {identifier_type} (e.g. 'my-value'). Got value {token.value}", error_code=INVALID_PARAMETER_VALUE)
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                if key != 'run_id':
                    raise MlflowException("Only the 'run_id' attribute supports comparison with a list of quoted string values.", error_code=INVALID_PARAMETER_VALUE)
                return cls._parse_run_ids(token)
            else:
                raise MlflowException(f'Expected a quoted string value or a list of quoted string values for attributes. Got value {token.value}', error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException(f'Invalid identifier type. Expected one of {[cls._ATTRIBUTE_IDENTIFIER, cls._TAG_IDENTIFIER]}.', error_code=INVALID_PARAMETER_VALUE)

    @classmethod
    def _invalid_statement_token_search_model_registry(cls, token):
        if isinstance(token, Comparison) or token.is_whitespace or token.match(ttype=TokenType.Keyword, values=['AND']):
            return False
        return True