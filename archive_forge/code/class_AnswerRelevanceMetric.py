from dataclasses import dataclass, field
from typing import Any, Dict, List
from mlflow.metrics.genai.base import EvaluationExample
from mlflow.metrics.genai.prompt_template import PromptTemplate
@dataclass
class AnswerRelevanceMetric:
    definition = 'Answer relevance measures the appropriateness and applicability of the output with respect to the input. Scores should reflect the extent to which the output directly addresses the question provided in the input, and give lower scores for incomplete or redundant output.'
    grading_prompt = "Answer relevance: Please give a score from 1-5 based on the degree of relevance to the input, where the lowest and highest scores are defined as follows:- Score 1: The output doesn't mention anything about the question or is completely irrelevant to the input.\n- Score 5: The output addresses all aspects of the question and all parts of the output are meaningful and relevant to the question."
    parameters = default_parameters
    default_model = default_model
    example_score_2 = EvaluationExample(input='How is MLflow related to Databricks?', output='Databricks is a company that specializes in big data and machine learning solutions.', score=2, justification='The output provided by the model does give some information about Databricks, which is part of the input question. However, it does not address the main point of the question, which is the relationship between MLflow and Databricks. Therefore, while the output is not completely irrelevant, it does not fully answer the question, leading to a lower score.')
    example_score_5 = EvaluationExample(input='How is MLflow related to Databricks?', output='MLflow is a product created by Databricks to enhance the efficiency of machine learning processes.', score=5, justification='The output directly addresses the input question by explaining the relationship between MLflow and Databricks. It provides a clear and concise answer that MLflow is a product created by Databricks, and also adds relevant information about the purpose of MLflow, which is to enhance the efficiency of machine learning processes. Therefore, the output is highly relevant to the input and deserves a full score.')
    default_examples = [example_score_2, example_score_5]