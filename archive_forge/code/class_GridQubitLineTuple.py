from typing import List, Iterable
import cirq
class GridQubitLineTuple(tuple):
    """A contiguous non-overlapping sequence of adjacent grid qubits."""

    @staticmethod
    def best_of(lines: Iterable[LineSequence], length: int) -> 'GridQubitLineTuple':
        lines = list(lines)
        longest = max(lines, key=len) if lines else []
        if len(longest) < length:
            raise NotFoundError('No line placement with desired length found.')
        return GridQubitLineTuple(longest[:length])

    def __str__(self) -> str:
        diagram = cirq.TextDiagramDrawer()
        dx = min((q.col for q in self))
        dy = min((q.row for q in self))
        for q in self:
            diagram.write(q.col - dx, q.row - dy, str(q))
        for q1, q2 in zip(self, self[1:]):
            diagram.grid_line(q1.col - dx, q1.row - dy, q2.col - dx, q2.row - dy, True)
        return diagram.render(horizontal_spacing=2, vertical_spacing=1, use_unicode_characters=True)