from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class ComparisonOperation(ABC):

    def __init__(self, field_name: str, comparison_value: Union[str, int, float, bool, List]):
        self.field_name = field_name
        self.comparison_value = comparison_value

    @abstractmethod
    def evaluate(self, fields) -> bool:
        pass

    @classmethod
    def parse(cls, field_name, comparison_clause: Union[Dict, List, str, float]) -> List['ComparisonOperation']:
        comparison_operations: List[ComparisonOperation] = []
        if isinstance(comparison_clause, dict):
            for comparison_operation, comparison_value in comparison_clause.items():
                if comparison_operation == '$eq':
                    comparison_operations.append(EqOperation(field_name, comparison_value))
                elif comparison_operation == '$in':
                    comparison_operations.append(InOperation(field_name, comparison_value))
                elif comparison_operation == '$ne':
                    comparison_operations.append(NeOperation(field_name, comparison_value))
                elif comparison_operation == '$nin':
                    comparison_operations.append(NinOperation(field_name, comparison_value))
                elif comparison_operation == '$gt':
                    comparison_operations.append(GtOperation(field_name, comparison_value))
                elif comparison_operation == '$gte':
                    comparison_operations.append(GteOperation(field_name, comparison_value))
                elif comparison_operation == '$lt':
                    comparison_operations.append(LtOperation(field_name, comparison_value))
                elif comparison_operation == '$lte':
                    comparison_operations.append(LteOperation(field_name, comparison_value))
        elif isinstance(comparison_clause, list):
            comparison_operations.append(InOperation(field_name, comparison_clause))
        else:
            comparison_operations.append(EqOperation(field_name, comparison_clause))
        return comparison_operations

    @abstractmethod
    def convert_to_elasticsearch(self):
        """
        Converts the ComparisonOperation instance to an Elasticsearch query.
        """
        pass

    @abstractmethod
    def convert_to_sql(self, meta_document_orm):
        """
        Converts the ComparisonOperation instance to an SQL filter.
        """
        pass

    @abstractmethod
    def convert_to_weaviate(self):
        """
        Converts the ComparisonOperation instance to a Weaviate comparison operator.
        """
        pass

    def convert_to_pinecone(self):
        """
        Converts the ComparisonOperation instance to a Pinecone comparison operator.
        """
        pass

    @abstractmethod
    def invert(self) -> 'ComparisonOperation':
        """
        Inverts the ComparisonOperation.
        Necessary for Weaviate as Weaviate doesn't seem to support the 'Not' operator anymore.
        (https://github.com/semi-technologies/weaviate/issues/1717)
        """
        pass

    def _get_weaviate_datatype(self, value: Optional[Union[str, int, float, bool]]=None) -> Tuple[str, Union[str, int, float, bool]]:
        """
        Determines the type of the comparison value and converts it to RFC3339 format if it is as date,
        as Weaviate requires dates to be in RFC3339 format including the time and timezone.

        """
        if value is None:
            assert not isinstance(self.comparison_value, list)
            value = self.comparison_value
        if isinstance(value, str):
            try:
                value = convert_date_to_rfc3339(value)
                data_type = 'valueDate'
            except ValueError:
                data_type = 'valueText' if self.field_name == 'content' else 'valueString'
        elif isinstance(value, int):
            data_type = 'valueInt'
        elif isinstance(value, float):
            data_type = 'valueNumber'
        elif isinstance(value, bool):
            data_type = 'valueBoolean'
        else:
            raise ValueError(f'Unsupported data type of comparison value for {self.__class__.__name__}.Value needs to be of type str, int, float, or bool.')
        return (data_type, value)