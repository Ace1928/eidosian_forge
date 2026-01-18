from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class NotOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'NOT' operations.
    """

    def evaluate(self, fields) -> bool:
        return not any((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'must_not': conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(~or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        conditions = [condition.invert().convert_to_weaviate() for condition in self.conditions]
        if len(conditions) > 1:
            return {'operator': 'Or', 'operands': conditions}
        else:
            return conditions[0]

    def convert_to_pinecone(self) -> Dict[str, Union[str, int, float, bool, List[Dict]]]:
        conditions = [condition.invert().convert_to_pinecone() for condition in self.conditions]
        return {'$or': conditions} if len(conditions) > 1 else conditions[0]

    def invert(self) -> Union[LogicalFilterClause, ComparisonOperation]:
        if len(self.conditions) > 1:
            return AndOperation(self.conditions)
        else:
            return self.conditions[0]