from typing import Callable, List, Optional, Tuple, Set, Any, TYPE_CHECKING
import numpy as np
import cirq
from cirq_google.line.placement import place_strategy, optimization
from cirq_google.line.placement.chip import above, right_of, chip_as_adjacency_list, EDGE
from cirq_google.line.placement.sequence import GridQubitLineTuple, LineSequence
class AnnealSequenceSearchStrategy(place_strategy.LinePlacementStrategy):
    """Linearized sequence search using simulated annealing method.

    TODO: This line search strategy is still work in progress and requires
    efficiency improvements.
    Github issue: https://github.com/quantumlib/Cirq/issues/2217
    """

    def __init__(self, trace_func: Optional[Callable[[List[LineSequence], float, float, float, bool], None]]=None, seed: Optional[int]=None) -> None:
        """Linearized sequence search using simulated annealing method.

        Args:
            trace_func: Optional callable which will be called for each
                        simulated annealing step with arguments: solution
                        candidate (list of linear sequences on the chip),
                        current temperature (float), candidate cost (float),
                        probability of accepting candidate (float), and
                        acceptance decision (boolean).
            seed: Optional seed value for random number generator.

        Returns:
            List of linear sequences on the chip found by simulated annealing
            method.
        """
        self.trace_func = trace_func
        self.seed = seed

    def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
        """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            List of linear sequences on the chip found by simulated annealing
            method.
        """
        seqs = AnnealSequenceSearch(device, self.seed).search(self.trace_func)
        return GridQubitLineTuple.best_of(seqs, length)