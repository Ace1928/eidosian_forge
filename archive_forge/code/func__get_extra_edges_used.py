import itertools
from qiskit.transpiler import CouplingMap, Target, AnalysisPass, TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit._accelerate.error_map import ErrorMap
def _get_extra_edges_used(self, dag, layout):
    """Returns the set of extra edges involved in the layout."""
    extra_edges_used = set()
    virtual_bits = layout.get_virtual_bits()
    for node in dag.two_qubit_ops():
        p0 = virtual_bits[node.qargs[0]]
        p1 = virtual_bits[node.qargs[1]]
        if self.coupling_map.distance(p0, p1) > 1:
            extra_edge = (p0, p1) if p0 < p1 else (p1, p0)
            extra_edges_used.add(extra_edge)
    return extra_edges_used