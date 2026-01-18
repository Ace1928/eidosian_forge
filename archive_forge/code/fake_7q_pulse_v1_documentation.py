import os
from qiskit.providers.fake_provider import fake_pulse_backend
A fake **pulse** backend with the following characteristics:

    * num_qubits: 7
    * coupling_map:

        .. code-block:: text

            0 ↔ 1 ↔ 3 ↔ 5 ↔ 6
                ↕       ↕
                2       4

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'u3', 'id', 'measure', 'u2', 'x', 'u1', 'sx', 'rz'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, 1, 2, 3, 4, 5, 6)
    