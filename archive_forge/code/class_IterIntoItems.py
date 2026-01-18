from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
class IterIntoItems:

    def __iter__(self):
        nonlocal done
        i = 0
        while True:
            if i == len(items) and (not done):
                try:
                    items.append(next(iterator))
                except StopIteration:
                    done = True
            if i < len(items):
                yield items[i]
                i += 1
            elif done:
                break