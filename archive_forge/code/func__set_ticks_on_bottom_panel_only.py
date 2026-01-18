from mplfinance._helpers import _list_of_dict
from mplfinance._arg_validators import _valid_panel_id
import pandas as pd
def _set_ticks_on_bottom_panel_only(panels, formatter, rotation=45, xlabel=None):
    bot = panels.index.values[-1]
    ax = panels.at[bot, 'axes'][0]
    ax.tick_params(axis='x', rotation=rotation)
    ax.xaxis.set_major_formatter(formatter)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if len(panels) == 1:
        return
    for panid in panels.index.values[::-1][1:]:
        panels.at[panid, 'axes'][0].tick_params(axis='x', labelbottom=False)