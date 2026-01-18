from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class LogicalFilterClause(ABC):
    """
    Class that is able to parse a filter and convert it to the format that the underlying databases of our
    DocumentStores require.

    Filters are defined as nested dictionaries. The keys of the dictionaries can be a logical
    operator (`"$and"`, `"$or"`, `"$not"`), a comparison operator (`"$eq"`, `"$in"`, `"$gt"`, `"$gte"`, `"$lt"`,
    `"$lte"`) or a metadata field name.
    Logical operator keys take a dictionary of metadata field names and/or logical operators as
    value. Metadata field names take a dictionary of comparison operators as value. Comparison
    operator keys take a single value or (in case of `"$in"`) a list of values as value.
    If no logical operator is provided, `"$and"` is used as default operation. If no comparison
    operator is provided, `"$eq"` (or `"$in"` if the comparison value is a list) is used as default
    operation.
    Example:
        ```python
        filters = {
            "$and": {
                "type": {"$eq": "article"},
                "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
                "rating": {"$gte": 3},
                "$or": {
                    "genre": {"$in": ["economy", "politics"]},
                    "publisher": {"$eq": "nytimes"}
                }
            }
        }
        # or simpler using default operators
        filters = {
            "type": "article",
            "date": {"$gte": "2015-01-01", "$lt": "2021-01-01"},
            "rating": {"$gte": 3},
            "$or": {
                "genre": ["economy", "politics"],
                "publisher": "nytimes"
            }
        }
        ```

    To use the same logical operator multiple times on the same level, logical operators take optionally a list of
    dictionaries as value.

    Example:
        ```python
        filters = {
            "$or": [
                {
                    "$and": {
                        "Type": "News Paper",
                        "Date": {
                            "$lt": "2019-01-01"
                        }
                    }
                },
                {
                    "$and": {
                        "Type": "Blog Post",
                        "Date": {
                            "$gte": "2019-01-01"
                        }
                    }
                }
            ]
        }
        ```

    """

    def __init__(self, conditions: List[Union['LogicalFilterClause', 'ComparisonOperation']]):
        self.conditions = conditions

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(cls, filter_term: Union[dict, List[dict]]) -> Union['LogicalFilterClause', 'ComparisonOperation']:
        """
        Parses a filter dictionary/list and returns a LogicalFilterClause instance.

        :param filter_term: Dictionary or list that contains the filter definition.
        """
        conditions: List[Union[LogicalFilterClause, ComparisonOperation]] = []
        if isinstance(filter_term, dict):
            filter_term = [filter_term]
        for item in filter_term:
            for key, value in item.items():
                if key == '$not':
                    conditions.append(NotOperation.parse(value))
                elif key == '$and':
                    conditions.append(AndOperation.parse(value))
                elif key == '$or':
                    conditions.append(OrOperation.parse(value))
                else:
                    conditions.extend(ComparisonOperation.parse(key, value))
        if cls == LogicalFilterClause:
            return conditions[0] if len(conditions) == 1 else AndOperation(conditions)
        else:
            return cls(conditions)

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the LogicalFilterClause instance to an Elasticsearch filter.
        """
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        """
        Converts the LogicalFilterClause instance to an SQL filter.
        """
        pass

    def convert_to_weaviate(self):
        """
        Converts the LogicalFilterClause instance to a Weaviate filter.
        """
        pass

    def convert_to_pinecone(self):
        """
        Converts the LogicalFilterClause instance to a Pinecone filter.
        """
        pass

    def _merge_es_range_queries(self, conditions: List[Dict]) -> List[Dict[str, Dict]]:
        """
        Merges Elasticsearch range queries that perform on the same metadata field.
        """
        range_conditions = [cond['range'] for cond in filter(lambda condition: 'range' in condition, conditions)]
        if range_conditions:
            conditions = [condition for condition in conditions if 'range' not in condition]
            range_conditions_dict = nested_defaultdict()
            for condition in range_conditions:
                field_name = list(condition.keys())[0]
                operation = list(condition[field_name].keys())[0]
                comparison_value = condition[field_name][operation]
                range_conditions_dict[field_name][operation] = comparison_value
            conditions.extend(({'range': {field_name: comparison_operations}} for field_name, comparison_operations in range_conditions_dict.items()))
        return conditions

    @abstractmethod
    def invert(self) -> Union['LogicalFilterClause', 'ComparisonOperation']:
        """
        Inverts the LogicalOperation instance.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """
        pass