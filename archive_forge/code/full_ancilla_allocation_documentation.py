from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target

        Checks if all the qregs in ``layout_qregs`` already exist in ``dag_qregs``. Otherwise, raise.
        