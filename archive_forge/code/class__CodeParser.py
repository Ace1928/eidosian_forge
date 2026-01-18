import re
from typing import Callable, List
import torch
from torch import Tensor
class _CodeParser:

    def __init__(self, code_string: str):
        optional_ws = '\\s*'
        required_ws = '\\s+'
        template_params = '(?P<template_params>\\<.+\\>)'
        return_type = '(?P<return_type>\\w+)'
        function_name = '(?P<function_name>\\w+)'
        function_params = '(?P<function_params>\\(.+\\))'
        function_body = '(?P<function_body>\\{.+\\})'
        pattern = optional_ws + 'template' + optional_ws + template_params + optional_ws + return_type + required_ws + function_name + optional_ws + function_params + optional_ws + function_body + optional_ws
        result = re.match(pattern, code_string, re.DOTALL)
        if result is None:
            raise Exception(f"Couldn't parse code, please check correctness:\n {code_string}")
        self.template_params = result['template_params']
        self.return_type = result['return_type']
        self.function_name = result['function_name']
        self.function_params = result['function_params']
        self.function_body = result['function_body']