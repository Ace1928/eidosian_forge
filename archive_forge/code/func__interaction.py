from typing import Iterator
import networkx as nx
import cirq
def _interaction(row_start_offset=0, row_end_offset=0, row_step=1, col_start_offset=0, col_end_offset=0, col_step=1, get_neighbor=lambda row, col: (row, col)):
    for row in range(row_start + row_start_offset, row_end + row_end_offset, row_step):
        for col in range(col_start + col_start_offset, col_end + col_end_offset, col_step):
            node1 = (row, col)
            if node1 not in problem_graph.nodes:
                continue
            node2 = get_neighbor(row, col)
            if node2 not in problem_graph.nodes:
                continue
            if (node1, node2) not in problem_graph.edges:
                continue
            weight = problem_graph.edges[node1, node2].get('weight', 1)
            yield two_qubit_gate(exponent=weight, global_shift=-0.5).on(cirq.GridQubit(*node1), cirq.GridQubit(*node2))