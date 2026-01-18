import dataclasses
from dataclasses import field
from types import CodeType, ModuleType
from typing import Any, Dict
@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    globals: Dict[str, Any] = field(default_factory=dict)
    locals: Dict[str, Any] = field(default_factory=dict)
    builtins: Dict[str, Any] = field(default_factory=dict)
    code_options: Dict[str, Any] = field(default_factory=dict)

    def dump(self, f):
        assert dill is not None, 'replay_record requires `pip install dill`'
        dill.dump(self, f)

    @classmethod
    def load(cls, f):
        assert dill is not None, 'replay_record requires `pip install dill`'
        return dill.load(f)