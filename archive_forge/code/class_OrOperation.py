from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, List, Dict, Optional, Tuple, Any
from .utils import convert_date_to_rfc3339
class OrOperation(LogicalFilterClause):
    """
    Handles conversion of logical 'OR' operations.
    """

    def evaluate(self, fields) -> bool:
        return any((condition.evaluate(fields) for condition in self.conditions))

    def convert_to_elasticsearch(self) -> Dict[str, Dict]:
        conditions = [condition.convert_to_elasticsearch() for condition in self.conditions]
        conditions = self._merge_es_range_queries(conditions)
        return {'bool': {'should': conditions}}

    def convert_to_sql(self, meta_document_orm):
        conditions = [meta_document_orm.document_id.in_(condition.convert_to_sql(meta_document_orm)) for condition in self.conditions]
        return select(meta_document_orm.document_id).filter(or_(*conditions))

    def convert_to_weaviate(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_weaviate() for condition in self.conditions]
        return {'operator': 'Or', 'operands': conditions}

    def convert_to_pinecone(self) -> Dict[str, Union[str, List[Dict]]]:
        conditions = [condition.convert_to_pinecone() for condition in self.conditions]
        return {'$or': conditions}

    def invert(self) -> AndOperation:
        return AndOperation([condition.invert() for condition in self.conditions])