import copy
from dataclasses import astuple, dataclass
from typing import (
import matplotlib as mpl
import matplotlib.collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1
from cirq.devices import grid_qubit
from cirq.vis import vis_utils
def _get_annotation_value(self, key, value) -> Optional[str]:
    if self._config.get('annotation_map'):
        return self._config['annotation_map'].get(key)
    elif self._config.get('annotation_format'):
        try:
            return format(value, self._config['annotation_format'])
        except:
            return format(float(value), self._config['annotation_format'])
    else:
        return None