import platform
import sys
from importlib.metadata import version
def _panel_comms():
    import panel as pn
    print(f'{'Panel comms':20}:  {pn.config.comms}')