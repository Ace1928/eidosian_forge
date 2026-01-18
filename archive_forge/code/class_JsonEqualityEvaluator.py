import json
from operator import eq
from typing import Any, Callable, Optional, Union, cast
from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown
class JsonEqualityEvaluator(StringEvaluator):
    """Evaluates whether the prediction is equal to the reference after
        parsing both as JSON.

    This evaluator checks if the prediction, after parsing as JSON, is equal
        to the reference,
    which is also parsed as JSON. It does not require an input string.

    Attributes:
        requires_input (bool): Whether this evaluator requires an
            input string. Always False.
        requires_reference (bool): Whether this evaluator requires
            a reference string. Always True.
        evaluation_name (str): The name of the evaluation metric.
            Always "parsed_equality".

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x['a'] == y['a'])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, operator: Optional[Callable]=None, **kwargs: Any) -> None:
        super().__init__()
        self.operator = operator or eq

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return 'json_equality'

    def _parse_json(self, string: Any) -> Union[dict, list, None, float, bool, int, str]:
        if isinstance(string, str):
            return parse_json_markdown(string)
        return string

    def _evaluate_strings(self, prediction: str, input: Optional[str]=None, reference: Optional[str]=None, **kwargs: Any) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator.
            reference (str): The reference string to compare against.

        Returns:
            dict: A dictionary containing the evaluation score.
        """
        parsed = self._parse_json(prediction)
        label = self._parse_json(cast(str, reference))
        if isinstance(label, list):
            if not isinstance(parsed, list):
                return {'score': 0}
            parsed = sorted(parsed, key=lambda x: str(x))
            label = sorted(label, key=lambda x: str(x))
        return {'score': self.operator(parsed, label)}