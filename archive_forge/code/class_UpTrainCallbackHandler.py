import logging
import sys
from collections import defaultdict
from typing import (
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
class UpTrainCallbackHandler(BaseCallbackHandler):
    """Callback Handler that logs evaluation results to uptrain and the console.

    Args:
        project_name_prefix (str): Prefix for the project name.
        key_type (str): Type of key to use. Must be 'uptrain' or 'openai'.
        api_key (str): API key for the UpTrain or OpenAI API.
        (This key is required to perform evaluations using GPT.)

    Raises:
        ValueError: If the key type is invalid.
        ImportError: If the `uptrain` package is not installed.

    """

    def __init__(self, *, project_name_prefix: str='langchain', key_type: str='openai', api_key: str='sk-****************', model: str='gpt-3.5-turbo', log_results: bool=True) -> None:
        """Initializes the `UpTrainCallbackHandler`."""
        super().__init__()
        uptrain = import_uptrain()
        self.log_results = log_results
        self.schema = UpTrainDataSchema(project_name_prefix=project_name_prefix)
        self.first_score_printed_flag = False
        if key_type == 'uptrain':
            settings = uptrain.Settings(uptrain_access_token=api_key, model=model)
            self.uptrain_client = uptrain.APIClient(settings=settings)
        elif key_type == 'openai':
            settings = uptrain.Settings(openai_api_key=api_key, evaluate_locally=False, model=model)
            self.uptrain_client = uptrain.EvalLLM(settings=settings)
        else:
            raise ValueError("Invalid key type: Must be 'uptrain' or 'openai'")

    def uptrain_evaluate(self, project_name: str, data: List[Dict[str, Any]], checks: List[str]) -> None:
        """Run an evaluation on the UpTrain server using UpTrain client."""
        if self.uptrain_client.__class__.__name__ == 'APIClient':
            uptrain_result = self.uptrain_client.log_and_evaluate(project_name=project_name, data=data, checks=checks)
        else:
            uptrain_result = self.uptrain_client.evaluate(data=data, checks=checks)
        self.schema.uptrain_results[project_name].append(uptrain_result)
        score_name_map = {'score_context_relevance': 'Context Relevance Score', 'score_factual_accuracy': 'Factual Accuracy Score', 'score_response_completeness': 'Response Completeness Score', 'score_sub_query_completeness': 'Sub Query Completeness Score', 'score_context_reranking': 'Context Reranking Score', 'score_context_conciseness': 'Context Conciseness Score', 'score_multi_query_accuracy': 'Multi Query Accuracy Score'}
        if self.log_results:
            logger.setLevel(logging.INFO)
        for row in uptrain_result:
            columns = list(row.keys())
            for column in columns:
                if column == 'question':
                    logger.info(f'\nQuestion: {row[column]}')
                    self.first_score_printed_flag = False
                elif column == 'response':
                    logger.info(f'Response: {row[column]}')
                    self.first_score_printed_flag = False
                elif column == 'variants':
                    logger.info('Multi Queries:')
                    for variant in row[column]:
                        logger.info(f'  - {variant}')
                    self.first_score_printed_flag = False
                elif column.startswith('score'):
                    if not self.first_score_printed_flag:
                        logger.info('')
                        self.first_score_printed_flag = True
                    if column in score_name_map:
                        logger.info(f'{score_name_map[column]}: {row[column]}')
                    else:
                        logger.info(f'{column}: {row[column]}')
        if self.log_results:
            logger.setLevel(logging.WARNING)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> None:
        """Log records to uptrain when an LLM ends."""
        uptrain = import_uptrain()
        self.schema.response = response.generations[0][0].text
        if 'qa_rag' in self.schema.eval_types and parent_run_id != self.schema.multi_query_daugher_run_id:
            data = [{'question': self.schema.query, 'context': self.schema.context, 'response': self.schema.response}]
            self.uptrain_evaluate(project_name=f'{self.schema.project_name_prefix}_rag', data=data, checks=[uptrain.Evals.CONTEXT_RELEVANCE, uptrain.Evals.FACTUAL_ACCURACY, uptrain.Evals.RESPONSE_COMPLETENESS])

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, tags: Optional[List[str]]=None, parent_run_id: Optional[UUID]=None, metadata: Optional[Dict[str, Any]]=None, run_type: Optional[str]=None, name: Optional[str]=None, **kwargs: Any) -> None:
        """Do nothing when chain starts"""
        if parent_run_id == self.schema.multi_query_run_id:
            self.schema.multi_query_daugher_run_id = run_id
        if isinstance(inputs, dict) and set(inputs.keys()) == {'context', 'question'}:
            self.schema.eval_types.add('qa_rag')
            context = ''
            if isinstance(inputs['context'], Document):
                context = inputs['context'].page_content
            elif isinstance(inputs['context'], list):
                for doc in inputs['context']:
                    context += doc.page_content + '\n'
            elif isinstance(inputs['context'], str):
                context = inputs['context']
            self.schema.context = context
            self.schema.query = inputs['question']
        pass

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID]=None, tags: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None, **kwargs: Any) -> None:
        if 'contextual_compression' in serialized['id']:
            self.schema.eval_types.add('contextual_compression')
            self.schema.query = query
            self.schema.context_conciseness_run_id = run_id
        if 'multi_query' in serialized['id']:
            self.schema.eval_types.add('multi_query')
            self.schema.multi_query_run_id = run_id
            self.schema.query = query
        elif 'multi_query' in self.schema.eval_types:
            self.schema.multi_queries.append(query)

    def on_retriever_end(self, documents: Sequence[Document], *, run_id: UUID, parent_run_id: Optional[UUID]=None, **kwargs: Any) -> Any:
        """Run when Retriever ends running."""
        uptrain = import_uptrain()
        if run_id == self.schema.multi_query_run_id:
            data = [{'question': self.schema.query, 'variants': self.schema.multi_queries}]
            self.uptrain_evaluate(project_name=f'{self.schema.project_name_prefix}_multi_query', data=data, checks=[uptrain.Evals.MULTI_QUERY_ACCURACY])
        if 'contextual_compression' in self.schema.eval_types:
            if parent_run_id == self.schema.context_conciseness_run_id:
                for doc in documents:
                    self.schema.old_context.append(doc.page_content)
            elif run_id == self.schema.context_conciseness_run_id:
                for doc in documents:
                    self.schema.new_context.append(doc.page_content)
                context = '\n'.join([f'{index}. {string}' for index, string in enumerate(self.schema.old_context, start=1)])
                reranked_context = '\n'.join([f'{index}. {string}' for index, string in enumerate(self.schema.new_context, start=1)])
                data = [{'question': self.schema.query, 'context': context, 'concise_context': reranked_context, 'reranked_context': reranked_context}]
                self.uptrain_evaluate(project_name=f'{self.schema.project_name_prefix}_context_reranking', data=data, checks=[uptrain.Evals.CONTEXT_CONCISENESS, uptrain.Evals.CONTEXT_RERANKING])