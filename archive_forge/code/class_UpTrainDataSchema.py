import logging
import sys
from collections import defaultdict
from typing import (
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
class UpTrainDataSchema:
    """The UpTrain data schema for tracking evaluation results.

    Args:
        project_name_prefix (str): Prefix for the project name.

    Attributes:
        project_name_prefix (str): Prefix for the project name.
        uptrain_results (DefaultDict[str, Any]): Dictionary to store evaluation results.
        eval_types (Set[str]): Set to store the types of evaluations.
        query (str): Query for the RAG evaluation.
        context (str): Context for the RAG evaluation.
        response (str): Response for the RAG evaluation.
        old_context (List[str]): Old context nodes for Context Conciseness evaluation.
        new_context (List[str]): New context nodes for Context Conciseness evaluation.
        context_conciseness_run_id (str): Run ID for Context Conciseness evaluation.
        multi_queries (List[str]): List of multi queries for Multi Query evaluation.
        multi_query_run_id (str): Run ID for Multi Query evaluation.
        multi_query_daugher_run_id (str): Run ID for Multi Query daughter evaluation.

    """

    def __init__(self, project_name_prefix: str) -> None:
        """Initialize the UpTrain data schema."""
        self.project_name_prefix: str = project_name_prefix
        self.uptrain_results: DefaultDict[str, Any] = defaultdict(list)
        self.eval_types: Set[str] = set()
        self.query: str = ''
        self.context: str = ''
        self.response: str = ''
        self.old_context: List[str] = []
        self.new_context: List[str] = []
        self.context_conciseness_run_id: UUID = UUID(int=0)
        self.multi_queries: List[str] = []
        self.multi_query_run_id: UUID = UUID(int=0)
        self.multi_query_daugher_run_id: UUID = UUID(int=0)