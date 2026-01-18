import math
import warnings
from typing import Optional
import torch
from torchaudio.functional.functional import _create_triangular_filterbank
def _hz_to_bark(freqs: float, bark_scale: str='traunmuller') -> float:
    """Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """
    if bark_scale not in ['schroeder', 'traunmuller', 'wang']:
        raise ValueError('bark_scale should be one of "schroeder", "traunmuller" or "wang".')
    if bark_scale == 'wang':
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == 'schroeder':
        return 7.0 * math.asinh(freqs / 650.0)
    barks = 26.81 * freqs / (1960.0 + freqs) - 0.53
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)
    return barks