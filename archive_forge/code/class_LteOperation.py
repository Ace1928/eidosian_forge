from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class LteOperation(ComparisonOperation):
    """
    Handles conversion of the '$lte' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] <= self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Union[str, float, int]]]]:
        assert not isinstance(self.comparison_value, list), "Comparison value for '$lte' operation must not be a list."
        return {'range': {self.field_name: {'lte': self.comparison_value}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value <= self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, float, int]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        assert not isinstance(comp_value, list), "Comparison value for '$lte' operation must not be a list."
        return {'path': [self.field_name], 'operator': 'LessThanEqual', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[float, int]]]:
        assert not isinstance(self.comparison_value, (list, str)), "Comparison value for '$lte' operation must be a float or int."
        return {self.field_name: {'$lte': self.comparison_value}}

    def invert(self) -> 'GtOperation':
        return GtOperation(self.field_name, self.comparison_value)