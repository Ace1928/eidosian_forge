import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _ask_options(input_text, options=[], convert_value=None, default=0):
    menu = BulletMenu(input_text, options)
    result = menu.run(default_choice=default)
    return convert_value(result) if convert_value is not None else result