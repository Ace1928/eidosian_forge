import re
import ast
from ochat.evaluation.grading.math_grader import grade_answer
def _function_exists(code, func_name):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return True
    return False