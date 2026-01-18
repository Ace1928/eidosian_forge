from __future__ import annotations
import dataclasses
import enum
import logging
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal.diagnostics.infra import formatter, sarif
@dataclasses.dataclass
class RuleCollection:
    _rule_id_name_set: FrozenSet[Tuple[str, str]] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._rule_id_name_set = frozenset({(field.default.id, field.default.name) for field in dataclasses.fields(self) if isinstance(field.default, Rule)})

    def __contains__(self, rule: Rule) -> bool:
        """Checks if the rule is in the collection."""
        return (rule.id, rule.name) in self._rule_id_name_set

    @classmethod
    def custom_collection_from_list(cls, new_collection_class_name: str, rules: Sequence[Rule]) -> RuleCollection:
        """Creates a custom class inherited from RuleCollection with the list of rules."""
        return dataclasses.make_dataclass(new_collection_class_name, [(formatter.kebab_case_to_snake_case(rule.name), type(rule), dataclasses.field(default=rule)) for rule in rules], bases=(cls,))()