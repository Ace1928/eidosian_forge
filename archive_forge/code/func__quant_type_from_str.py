import enum
def _quant_type_from_str(name: str) -> QuantType:
    for quant_type, s in _quant_type_to_str.items():
        if name == s:
            return quant_type
    raise ValueError(f"Unknown QuantType name '{name}'")