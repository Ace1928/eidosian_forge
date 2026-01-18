from collections.abc import Callable, Sequence
from os import listdir, path
import numpy as np
from .functions import clip_array, clip_scalar, colorDistance, eq, mkColor
from .Qt import QtCore, QtGui
def _getFromFile(name):
    filename = name
    if filename[0] != '.':
        dirname = path.dirname(__file__)
        filename = path.join(dirname, 'colors/maps/' + filename)
    if not path.isfile(filename):
        if path.isfile(filename + '.csv'):
            filename += '.csv'
        elif path.isfile(filename + '.hex'):
            filename += '.hex'
    with open(filename, 'r') as fh:
        idx = 0
        color_list = []
        if filename[-4:].lower() != '.hex':
            csv_mode = True
        else:
            csv_mode = False
        for line in fh:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == ';':
                continue
            parts = line.split(sep=';', maxsplit=1)
            if csv_mode:
                comp = parts[0].split(',')
                if len(comp) < 3:
                    continue
                color_tuple = tuple([int(255 * float(c) + 0.5) for c in comp])
            else:
                hex_str = parts[0]
                if hex_str[0] == '#':
                    hex_str = hex_str[1:]
                if len(hex_str) < 3:
                    continue
                if len(hex_str) == 3:
                    hex_str = 2 * hex_str[0] + 2 * hex_str[1] + 2 * hex_str[2]
                elif len(hex_str) == 4:
                    hex_str = 2 * hex_str[0] + 2 * hex_str[1] + 2 * hex_str[2] + 2 * hex_str[3]
                if len(hex_str) < 6:
                    continue
                try:
                    color_tuple = tuple(bytes.fromhex(hex_str))
                except ValueError as e:
                    raise ValueError(f"failed to convert hexadecimal value '{hex_str}'.") from e
            color_list.append(color_tuple)
            idx += 1
    cmap = ColorMap(name=name, pos=np.linspace(0.0, 1.0, len(color_list)), color=color_list)
    if cmap is not None:
        cmap.name = name
        _mapCache[name] = cmap
    return cmap