from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class NeOperation(ComparisonOperation):
    """
    Handles conversion of the '$ne' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] != self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Union[str, int, float, bool]]]]]:
        assert not isinstance(self.comparison_value, list), "Use '$nin' operation for lists as comparison values."
        return {'bool': {'must_not': {'term': {self.field_name: self.comparison_value}}}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value != self.comparison_value)

    def convert_to_weaviate(self) -> Dict[str, Union[List[str], str, int, float, bool]]:
        comp_value_type, comp_value = self._get_weaviate_datatype()
        return {'path': [self.field_name], 'operator': 'NotEqual', comp_value_type: comp_value}

    def convert_to_pinecone(self) -> Dict[str, Dict[str, Union[List[str], str, int, float, bool]]]:
        return {self.field_name: {'$ne': self.comparison_value}}

    def invert(self) -> 'EqOperation':
        return EqOperation(self.field_name, self.comparison_value)