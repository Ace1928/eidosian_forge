from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def _freq_to_text(formatter: Dict[str, Any], freq: float, unit: str='MHz') -> Tuple[str, str]:
    """A helper function to convert a freq value to text with supplementary unit.

    Args:
        formatter: Dictionary of stylesheet settings.
        freq: A frequency value in units of Hz.
        unit: Supplementary unit. THz, GHz, MHz, kHz, Hz are supported.

    Returns:
        Standard text and latex text of phase value.

    Raises:
        VisualizationError: When unsupported unit is specified.
    """
    try:
        freq = float(freq)
    except TypeError:
        return (formatter['unicode_symbol.freq_parameter'], formatter['latex_symbol.freq_parameter'])
    unit_table = {'THz': 1000000000000.0, 'GHz': 1000000000.0, 'MHz': 1000000.0, 'kHz': 1000.0, 'Hz': 1}
    try:
        value = freq / unit_table[unit]
    except KeyError as ex:
        raise VisualizationError(f'Unit {unit} is not supported.') from ex
    latex = f'{value:.2f}~{{\\rm {unit}}}'
    plain = f'{value:.2f} {unit}'
    return (plain, latex)