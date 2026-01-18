import json
from typing import Any, Callable, Optional, Union
from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown
def _parse_json(self, node: Any) -> Union[dict, list, None, float, bool, int, str]:
    if isinstance(node, str):
        return parse_json_markdown(node)
    return node