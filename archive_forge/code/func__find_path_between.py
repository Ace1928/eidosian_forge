from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
def _find_path_between(self, p: GridQubit, q: GridQubit, used: Set[GridQubit]) -> Optional[List[GridQubit]]:
    """Searches for continuous sequence between two qubits.

        This method runs two BFS algorithms in parallel (alternating variable s
        in each iteration); the first one starting from qubit p, and the second
        one starting from qubit q. If at some point a qubit reachable from p is
        found to be on the set of qubits already reached from q (or vice versa),
        the search is stopped and new path returned.

        Args:
            p: The first qubit, start of the sequence.
            q: The second qubit, end of the sequence.
            used: Set of forbidden qubits which cannot appear on the sequence.

        Returns:
            Continues sequence of qubits with new path between p and q, or None
            if no path was found.
        """

    def assemble_path(n: GridQubit, parent: Dict[GridQubit, GridQubit]):
        path = [n]
        while n in parent:
            n = parent[n]
            path.append(n)
        return path
    other = {p: q, q: p}
    parents: Dict[GridQubit, Dict[GridQubit, GridQubit]] = {p: {}, q: {}}
    visited: Dict[GridQubit, Set[GridQubit]] = {p: set(), q: set()}
    queue = collections.deque([(p, p), (q, q)])
    while queue:
        n, s = queue.popleft()
        for n_adj in self._c_adj[n]:
            if n_adj in visited[other[s]]:
                path_s = assemble_path(n, parents[s])[-2::-1]
                path_other = assemble_path(n_adj, parents[other[s]])[:-1]
                path = path_s + path_other
                if s == q:
                    path.reverse()
                return path
            if n_adj not in used and n_adj not in visited[s]:
                queue.append((n_adj, s))
                visited[s].add(n_adj)
                parents[s][n_adj] = n
    return None