from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
class ConvertCustomConfig:
    """
    Custom configuration for :func:`~torch.ao.quantization.quantize_fx.convert_fx`.

    Example usage::

        convert_custom_config = ConvertCustomConfig()             .set_observed_to_quantized_mapping(ObservedCustomModule, QuantizedCustomModule)             .set_preserved_attributes(["attr1", "attr2"])
    """

    def __init__(self):
        self.observed_to_quantized_mapping: Dict[QuantType, Dict[Type, Type]] = {}
        self.preserved_attributes: List[str] = []

    def __repr__(self):
        dict_nonempty = {k: v for k, v in self.__dict__.items() if len(v) > 0}
        return f'ConvertCustomConfig({dict_nonempty})'

    def set_observed_to_quantized_mapping(self, observed_class: Type, quantized_class: Type, quant_type: QuantType=QuantType.STATIC) -> ConvertCustomConfig:
        """
        Set the mapping from a custom observed module class to a custom quantized module class.

        The quantized module class must have a ``from_observed`` class method that converts the observed module class
        to the quantized module class.
        """
        if quant_type not in self.observed_to_quantized_mapping:
            self.observed_to_quantized_mapping[quant_type] = {}
        self.observed_to_quantized_mapping[quant_type][observed_class] = quantized_class
        return self

    def set_preserved_attributes(self, attributes: List[str]) -> ConvertCustomConfig:
        """
        Set the names of the attributes that will persist in the graph module even if they are not used in
        the model's ``forward`` method.
        """
        self.preserved_attributes = attributes
        return self

    @classmethod
    def from_dict(cls, convert_custom_config_dict: Dict[str, Any]) -> ConvertCustomConfig:
        """
        Create a ``ConvertCustomConfig`` from a dictionary with the following items:

            "observed_to_quantized_custom_module_class": a nested dictionary mapping from quantization
            mode to an inner mapping from observed module classes to quantized module classes, e.g.::
            {
            "static": {FloatCustomModule: ObservedCustomModule},
            "dynamic": {FloatCustomModule: ObservedCustomModule},
            "weight_only": {FloatCustomModule: ObservedCustomModule}
            }
            "preserved_attributes": a list of attributes that persist even if they are not used in ``forward``

        This function is primarily for backward compatibility and may be removed in the future.
        """
        conf = cls()
        for quant_type_name, custom_module_mapping in convert_custom_config_dict.get(OBSERVED_TO_QUANTIZED_DICT_KEY, {}).items():
            quant_type = _quant_type_from_str(quant_type_name)
            for observed_class, quantized_class in custom_module_mapping.items():
                conf.set_observed_to_quantized_mapping(observed_class, quantized_class, quant_type)
        conf.set_preserved_attributes(convert_custom_config_dict.get(PRESERVED_ATTRIBUTES_DICT_KEY, []))
        return conf

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``ConvertCustomConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig.from_dict`.
        """
        d: Dict[str, Any] = {}
        for quant_type, observed_to_quantized_mapping in self.observed_to_quantized_mapping.items():
            if OBSERVED_TO_QUANTIZED_DICT_KEY not in d:
                d[OBSERVED_TO_QUANTIZED_DICT_KEY] = {}
            d[OBSERVED_TO_QUANTIZED_DICT_KEY][_get_quant_type_to_str(quant_type)] = observed_to_quantized_mapping
        if len(self.preserved_attributes) > 0:
            d[PRESERVED_ATTRIBUTES_DICT_KEY] = self.preserved_attributes
        return d