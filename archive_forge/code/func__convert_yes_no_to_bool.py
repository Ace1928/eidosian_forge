import argparse
from ...utils.dataclasses import (
from ..menu import BulletMenu
def _convert_yes_no_to_bool(value):
    return {'yes': True, 'no': False}[value.lower()]