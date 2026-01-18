import logging
import os
from typing import Dict, List, Optional
import ray._private.ray_constants as ray_constants
from ray._private.utils import (
def _format_ports(self, pre_selected_ports):
    """Format the pre-selected ports information to be more human-readable."""
    ports = pre_selected_ports.copy()
    for comp, port_list in ports.items():
        if len(port_list) == 1:
            ports[comp] = port_list[0]
        elif len(port_list) == 0:
            ports[comp] = 'random'
        elif comp == 'worker_ports':
            min_port = port_list[0]
            max_port = port_list[len(port_list) - 1]
            if len(port_list) < 50:
                port_range_str = str(port_list)
            else:
                port_range_str = f'from {min_port} to {max_port}'
            ports[comp] = f'{len(port_list)} ports {port_range_str}'
    return ports