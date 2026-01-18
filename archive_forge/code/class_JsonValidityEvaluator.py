import json
from operator import eq
from typing import Any, Callable, Optional, Union, cast
from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown
class JsonValidityEvaluator(StringEvaluator):
    """Evaluates whether the prediction is valid JSON.

    This evaluator checks if the prediction is a valid JSON string. It does not
        require any input or reference.

    Attributes:
        requires_input (bool): Whether this evaluator requires an input
            string. Always False.
        requires_reference (bool): Whether this evaluator requires a
            reference string. Always False.
        evaluation_name (str): The name of the evaluation metric.
            Always "json".

    Examples:
        >>> evaluator = JsonValidityEvaluator()
        >>> prediction = '{"name": "John", "age": 30, "city": "New York"}'
        >>> evaluator.evaluate(prediction)
        {'score': 1}

        >>> prediction = '{"name": "John", "age": 30, "city": "New York",}'
        >>> evaluator.evaluate(prediction)
        {'score': 0, 'reasoning': 'Expecting property name enclosed in double quotes'}
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return 'json_validity'

    def _evaluate_strings(self, prediction: str, input: Optional[str]=None, reference: Optional[str]=None, **kwargs: Any) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator. Defaults to None.
            reference (str, optional): Not used in this evaluator. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation score. The score is 1 if
            the prediction is valid JSON, and 0 otherwise.
                If the prediction is not valid JSON, the dictionary also contains
                a "reasoning" field with the error message.

        """
        try:
            parse_json_markdown(prediction, parser=json.loads)
            return {'score': 1}
        except Exception as e:
            return {'score': 0, 'reasoning': str(e)}