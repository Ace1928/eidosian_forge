import os
from qiskit.providers.fake_provider import fake_pulse_backend
A fake **pulse** backend with the following characteristics:

    * num_qubits: 27
    * coupling_map:

        .. code-block:: text

                           06                  17
                           ↕                    ↕
            00 ↔ 01 ↔ 04 ↔ 07 ↔ 10 ↔ 12 ↔ 15 ↔ 18 ↔ 20 ↔ 23
                 ↕                   ↕                    ↕
                 02                  13                  24
                 ↕                   ↕                    ↕
                 03 ↔ 05 ↔ 08 ↔ 11 ↔ 14 ↔ 16 ↔ 19 ↔ 22 ↔ 25 ↔ 26
                           ↕                    ↕
                           09                  20

    * basis_gates: ``["id", "rz", "sx", "x", "cx", "reset"]``
    * scheduled instructions:
        # ``{'id', 'rz', 'u2', 'x', 'u3', 'sx', 'measure', 'u1'}`` for all individual qubits
        # ``{'cx'}`` for all edges
        # ``{'measure'}`` for (0, ..., 26)
    