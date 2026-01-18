from typing import (
from cirq import circuits
from cirq.interop.quirk.cells.cell import Cell
def _iterator_to_iterable(iterator: Iterator[T]) -> Iterable[T]:
    done = False
    items: List[T] = []

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
    return IterIntoItems()