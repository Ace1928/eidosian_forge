from typing import Dict, Union, List
import dateutil.parser
def _decode_pulse_library_item(pulse_library_item: Dict) -> None:
    """Decode a pulse library item.

    Args:
        pulse_library_item: A ``PulseLibraryItem`` in dictionary format.
    """
    pulse_library_item['samples'] = [_to_complex(sample) for sample in pulse_library_item['samples']]