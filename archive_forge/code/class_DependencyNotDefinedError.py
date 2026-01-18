from typing import Iterable
class DependencyNotDefinedError(DependencyDefinitionError):

    def __init__(self, name: str, expected: Iterable[str], actual: Iterable[str]):
        s = f'expected {sorted(expected)}, actual {sorted(actual)}'
        super().__init__(f'Task {name} dependencies not well defined: ' + s)