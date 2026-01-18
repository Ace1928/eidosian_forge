from collections import deque
from typing import Callable, Deque, Iterable, List, Optional, Tuple
from .otBase import BaseTable
def _traverse_ot_data(root: BaseTable, root_accessor: Optional[str], skip_root: bool, predicate: Optional[Callable[[SubTablePath], bool]], add_to_frontier_fn: AddToFrontierFn, iter_subtables_fn: Optional[Callable[[BaseTable], Iterable[BaseTable.SubTableEntry]]]=None) -> Iterable[SubTablePath]:
    if root_accessor is None:
        root_accessor = type(root).__name__
    if predicate is None:

        def predicate(path):
            return True
    if iter_subtables_fn is None:

        def iter_subtables_fn(table):
            return table.iterSubTables()
    frontier: Deque[SubTablePath] = deque()
    root_entry = BaseTable.SubTableEntry(root_accessor, root)
    if not skip_root:
        frontier.append((root_entry,))
    else:
        add_to_frontier_fn(frontier, [(root_entry, subtable_entry) for subtable_entry in iter_subtables_fn(root)])
    while frontier:
        path = frontier.popleft()
        current = path[-1].value
        if not predicate(path):
            continue
        yield SubTablePath(path)
        new_entries = [path + (subtable_entry,) for subtable_entry in iter_subtables_fn(current)]
        add_to_frontier_fn(frontier, new_entries)