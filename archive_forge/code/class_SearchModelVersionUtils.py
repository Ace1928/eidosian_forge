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
class SearchModelVersionUtils(SearchUtils):
    NUMERIC_ATTRIBUTES = {'version_number', 'creation_timestamp', 'last_updated_timestamp'}
    VALID_SEARCH_ATTRIBUTE_KEYS = {'name', 'version_number', 'run_id', 'source_path'}
    VALID_ORDER_BY_ATTRIBUTE_KEYS = {'name', 'version_number', 'creation_timestamp', 'last_updated_timestamp'}
    VALID_STRING_ATTRIBUTE_COMPARATORS = {'!=', '=', 'LIKE', 'ILIKE', 'IN'}

    @classmethod
    def _does_model_version_match_clauses(cls, mv, sed):
        key_type = sed.get('type')
        key = sed.get('key')
        value = sed.get('value')
        comparator = sed.get('comparator').upper()
        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(mv, 'source' if key == 'source_path' else key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            if key == 'version_number':
                key = 'version'
            lhs = getattr(mv, key)
            value = int(value)
        elif cls.is_tag(key_type, comparator):
            lhs = mv.tags.get(key, None)
        else:
            raise MlflowException(f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE)
        if lhs is None:
            return False
        if comparator == 'IN' and isinstance(value, (set, list)):
            return lhs in set(value)
        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, model_versions, filter_string):
        """Filters a set of model versions based on a search filter string."""
        model_versions = [mv for mv in model_versions if mv.current_stage != STAGE_DELETED_INTERNAL]
        if not filter_string:
            return model_versions
        parsed = cls.parse_search_filter(filter_string)

        def model_version_matches(mv):
            return all((cls._does_model_version_match_clauses(mv, s) for s in parsed))
        return [mv for mv in model_versions if model_version_matches(mv)]

    @classmethod
    def parse_order_by_for_search_model_versions(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = SearchExperimentsUtils._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        return (identifier['type'], identifier['key'], is_ascending)

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_model_versions, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == 'attribute':
                if key == 'version_number':
                    key = 'version'
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f'Invalid order_by entity: {type_}')
        if not any((key == 'name' for key, _ in order_by)):
            order_by.append(('name', True))
        if not any((key == 'version_number' for key, _ in order_by)):
            order_by.append(('version', False))
        return lambda model_version: tuple((_apply_reversor(model_version, k, asc) for k, asc in order_by))

    @classmethod
    def sort(cls, model_versions, order_by_list):
        return sorted(model_versions, key=cls._get_sort_key(order_by_list))

    @classmethod
    def _get_model_version_search_identifier(cls, identifier, valid_attributes):
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
        comp = cls._get_model_version_search_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
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
            elif token.ttype in cls.NUMERIC_VALUE_TYPES:
                if key not in cls.NUMERIC_ATTRIBUTES:
                    raise MlflowException(f"Only the '{cls.NUMERIC_ATTRIBUTES}' attributes support comparison with numeric values.", error_code=INVALID_PARAMETER_VALUE)
                if token.ttype == TokenType.Literal.Number.Integer:
                    return int(token.value)
                elif token.ttype == TokenType.Literal.Number.Float:
                    return float(token.value)
            else:
                raise MlflowException(f'Expected a quoted string value or a list of quoted string values for attributes. Got value {token.value}', error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException(f'Invalid identifier type. Expected one of {[cls._ATTRIBUTE_IDENTIFIER, cls._TAG_IDENTIFIER]}.', error_code=INVALID_PARAMETER_VALUE)

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_model_version, tokens))
        if len(invalids) > 0:
            invalid_clauses = ', '.join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(f'Invalid clause(s) in filter string: {invalid_clauses}')
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _invalid_statement_token_search_model_version(cls, token):
        if isinstance(token, Comparison) or token.is_whitespace or token.match(ttype=TokenType.Keyword, values=['AND']):
            return False
        return True

    @classmethod
    def parse_search_filter(cls, filter_string):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(f"Error on parsing filter '{filter_string}'", error_code=INVALID_PARAMETER_VALUE)
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(f"Invalid filter '{filter_string}'. Could not be parsed.", error_code=INVALID_PARAMETER_VALUE)
        elif len(parsed) > 1:
            raise MlflowException("Search filter contained multiple expression '%s'. Provide AND-ed expression list." % filter_string, error_code=INVALID_PARAMETER_VALUE)
        return cls._process_statement(parsed[0])