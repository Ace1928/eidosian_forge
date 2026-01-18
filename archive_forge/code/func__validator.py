from __future__ import annotations
import logging # isort:skip
from typing import (
from .check import ValidationIssue, Validator, ValidatorType
from .issue import Error, Issue, Warning
def _validator(code_or_name: int | str | Issue, validator_type: ValidatorType) -> ValidationDecorator:
    """ Internal shared implementation to handle both error and warning
    validation checks.

    Args:
        code code_or_name (int, str or Issue) : a defined error code or custom message
        validator_type (str) : either "error" or "warning"

    Returns:
        validation decorator

    """
    issues: type[Error] | type[Warning] = Error if validator_type == 'error' else Warning

    def decorator(func: ValidationFunction) -> Validator:
        assert func.__name__.startswith('_check'), f"validation function {func.__qualname__} must have '_check' prefix"

        def _wrapper(*args: Any, **kwargs: Any) -> list[ValidationIssue]:
            extra = func(*args, **kwargs)
            if extra is None:
                return []
            issue: Issue
            name: str
            if isinstance(code_or_name, str):
                issue = issues.get_by_name('EXT')
                name = f'{issue.name}:{code_or_name}'
            elif isinstance(code_or_name, int):
                try:
                    issue = issues.get_by_code(code_or_name)
                    name = issue.name
                except KeyError:
                    raise ValueError(f'unknown {validator_type} code {code_or_name}')
            else:
                issue = code_or_name
                name = issue.name
            code = issue.code
            text = issue.description
            return [ValidationIssue(code, name, text, extra)]
        wrapper = cast(Validator, _wrapper)
        wrapper.validator_type = validator_type
        return wrapper
    return decorator