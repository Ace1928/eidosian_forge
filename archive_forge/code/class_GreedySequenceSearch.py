from typing import Dict, List, Optional, Set, TYPE_CHECKING
import abc
import collections
from cirq.devices import GridQubit
from cirq_google.line.placement import place_strategy
from cirq_google.line.placement.chip import chip_as_adjacency_list
from cirq_google.line.placement.sequence import GridQubitLineTuple
class GreedySequenceSearch:
    """Base class for greedy search heuristics.

    Specialized greedy heuristics should implement abstract _sequence_search
    method.
    """

    def __init__(self, device: 'cirq_google.GridDevice', start: GridQubit) -> None:
        """Greedy sequence search constructor.

        Args:
            device: Chip description.
            start: Starting qubit.

        Raises:
            ValueError: When start qubit is not part of a chip.
        """
        if start not in device.metadata.qubit_set:
            raise ValueError('Starting qubit must be a qubit on the chip')
        self._c = device.metadata.qubit_set
        self._c_adj = chip_as_adjacency_list(device)
        self._start = start
        self._sequence: Optional[List[GridQubit]] = None

    def get_or_search(self) -> List[GridQubit]:
        """Starts the search or gives previously calculated sequence.

        Returns:
            The linear qubit sequence found.
        """
        if not self._sequence:
            self._sequence = self._find_sequence()
        return self._sequence

    @abc.abstractmethod
    def _choose_next_qubit(self, qubit: GridQubit, used: Set[GridQubit]) -> Optional[GridQubit]:
        """Selects next qubit on the linear sequence.

        Args:
            qubit: Last qubit which is already present on the linear sequence
                   of qubits.
            used: Set of forbidden qubits which can not be used.

        Returns:
            Next qubit to be appended to the linear sequence, chosen according
            to the greedy heuristic method. The returned qubit will be the one
            passed to the next invocation of this method. Returns None if no
            more qubits are available and search should stop.
        """

    def _find_sequence(self) -> List[GridQubit]:
        """Looks for a sequence starting at a given qubit.

        Search is issued twice from the starting qubit, so that longest possible
        sequence is found. Starting qubit might not be the first qubit on the
        returned sequence.

        Returns:
            The longest sequence found by this method.
        """
        tail = self._sequence_search(self._start, [])
        tail.pop(0)
        head = self._sequence_search(self._start, tail)
        head.reverse()
        return self._expand_sequence(head + tail)

    def _sequence_search(self, start: GridQubit, current: List[GridQubit]) -> List[GridQubit]:
        """Search for the continuous linear sequence from the given qubit.

        This method is called twice for the same starting qubit, so that
        sequences that begin and end on this qubit are searched for.

        Args:
            start: The first qubit, where search should be triggered from.
            current: Previously found linear sequence, which qubits are
                     forbidden to use during the search.

        Returns:
            Continuous linear sequence that begins with the starting qubit and
            does not contain any qubits from the current list.
        """
        used = set(current)
        seq = []
        n: Optional[GridQubit] = start
        while n is not None:
            seq.append(n)
            used.add(n)
            n = self._choose_next_qubit(n, used)
        return seq

    def _expand_sequence(self, seq: List[GridQubit]) -> List[GridQubit]:
        """Tries to expand given sequence with more qubits.

        Args:
            seq: Linear sequence of qubits.

        Returns:
            New continuous linear sequence which contains all the qubits from
            seq and possibly new qubits inserted in between.
        """
        i = 1
        while i < len(seq):
            path = self._find_path_between(seq[i - 1], seq[i], set(seq))
            if path:
                seq = seq[:i] + path + seq[i:]
            else:
                i += 1
        return seq

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

    def _neighbors_of_excluding(self, qubit: GridQubit, used: Set[GridQubit]) -> List[GridQubit]:
        return [n for n in self._c_adj[qubit] if n not in used]