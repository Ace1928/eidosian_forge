from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Union
from warnings import warn
from langchain_core.agents import AgentAction
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.config import run_in_executor
from langchain.chains.base import Chain
class EvaluatorType(str, Enum):
    """The types of the evaluators."""
    QA = 'qa'
    'Question answering evaluator, which grades answers to questions\n    directly using an LLM.'
    COT_QA = 'cot_qa'
    "Chain of thought question answering evaluator, which grades\n    answers to questions using\n    chain of thought 'reasoning'."
    CONTEXT_QA = 'context_qa'
    "Question answering evaluator that incorporates 'context' in the response."
    PAIRWISE_STRING = 'pairwise_string'
    'The pairwise string evaluator, which predicts the preferred prediction from\n    between two models.'
    SCORE_STRING = 'score_string'
    'The scored string evaluator, which gives a score between 1 and 10 \n    to a prediction.'
    LABELED_PAIRWISE_STRING = 'labeled_pairwise_string'
    'The labeled pairwise string evaluator, which predicts the preferred prediction\n    from between two models based on a ground truth reference label.'
    LABELED_SCORE_STRING = 'labeled_score_string'
    'The labeled scored string evaluator, which gives a score between 1 and 10\n    to a prediction based on a ground truth reference label.'
    AGENT_TRAJECTORY = 'trajectory'
    "The agent trajectory evaluator, which grades the agent's intermediate steps."
    CRITERIA = 'criteria'
    'The criteria evaluator, which evaluates a model based on a\n    custom set of criteria without any reference labels.'
    LABELED_CRITERIA = 'labeled_criteria'
    'The labeled criteria evaluator, which evaluates a model based on a\n    custom set of criteria, with a reference label.'
    STRING_DISTANCE = 'string_distance'
    'Compare predictions to a reference answer using string edit distances.'
    EXACT_MATCH = 'exact_match'
    'Compare predictions to a reference answer using exact matching.'
    REGEX_MATCH = 'regex_match'
    'Compare predictions to a reference answer using regular expressions.'
    PAIRWISE_STRING_DISTANCE = 'pairwise_string_distance'
    'Compare predictions based on string edit distances.'
    EMBEDDING_DISTANCE = 'embedding_distance'
    'Compare a prediction to a reference label using embedding distance.'
    PAIRWISE_EMBEDDING_DISTANCE = 'pairwise_embedding_distance'
    'Compare two predictions using embedding distance.'
    JSON_VALIDITY = 'json_validity'
    'Check if a prediction is valid JSON.'
    JSON_EQUALITY = 'json_equality'
    'Check if a prediction is equal to a reference JSON.'
    JSON_EDIT_DISTANCE = 'json_edit_distance'
    'Compute the edit distance between two JSON strings after canonicalization.'
    JSON_SCHEMA_VALIDATION = 'json_schema_validation'
    'Check if a prediction is valid JSON according to a JSON schema.'