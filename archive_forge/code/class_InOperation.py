from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class InOperation(ComparisonOperation):
    """
    Handles conversion of the '$in' comparison operation.
    """

    def evaluate(self, fields) -> bool:
        if self.field_name not in fields:
            return False
        return fields[self.field_name] in self.comparison_value

    def convert_to_elasticsearch(self) -> Dict[str, Dict[str, List]]:
        assert isinstance(self.comparison_value, list), "'$in' operation requires comparison value to be a list."
        return {'terms': {self.field_name: self.comparison_value}}

    def convert_to_sql(self, meta_document_orm):
        return select([meta_document_orm.document_id]).where(meta_document_orm.name == self.field_name, meta_document_orm.value.in_(self.comparison_value))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        filter_dict: Dict[str, Union[str, List[Dict]]] = {'operator': 'Or', 'operands': []}
        assert isinstance(self.comparison_value, list), "'$in' operation requires comparison value to be a list."
        for value in self.comparison_value:
            comp_value_type, comp_value = self._get_weaviate_datatype(value)
            assert isinstance(filter_dict['operands'], list)
            filter_dict['operands'].append({'path': [self.field_name], 'operator': 'Equal', comp_value_type: comp_value})
        return filter_dict

    def convert_to_pinecone(self) -> Dict[str, Dict[str, List]]:
        assert isinstance(self.comparison_value, list), "'$in' operation requires comparison value to be a list."
        return {self.field_name: {'$in': self.comparison_value}}

    def invert(self) -> 'NinOperation':
        return NinOperation(self.field_name, self.comparison_value)