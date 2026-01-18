import os
from typing import Any, Callable, List, Optional, Tuple
import torch.utils.data as data
from ..utils import _log_api_usage_once
class StandardTransform:

    def __init__(self, transform: Optional[Callable]=None, target_transform: Optional[Callable]=None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (input, target)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f'{head}{lines[0]}'] + ['{}{}'.format(' ' * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, 'Transform: ')
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, 'Target transform: ')
        return '\n'.join(body)