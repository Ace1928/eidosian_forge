from typing import Iterable
import cirq
from cirq_web import widget
from cirq_web.circuits.symbols import (
class Circuit3D(widget.Widget):
    """Takes cirq.Circuit objects and displays them in 3D."""

    def __init__(self, circuit: cirq.Circuit, resolvers: Iterable[SymbolResolver]=DEFAULT_SYMBOL_RESOLVERS, padding_factor: float=1):
        """Initializes a Circuit instance.

        Args:
            circuit: The `cirq.Circuit` to be represented in 3D.
            resolvers: The symbol resolve for how to show symbols in 3D.
            padding_factor: The distance between meshes.
        """
        super().__init__()
        self.circuit = circuit
        self._resolvers = resolvers
        self.padding_factor = padding_factor

    def get_client_code(self) -> str:
        stripped_id = self.id.replace('-', '')
        moments = len(self.circuit.moments)
        self.serialized_circuit = self._serialize_circuit()
        return f'''\n            <button id="camera-reset">Reset Camera</button>\n            <button id="camera-toggle">Toggle Camera Type</button>\n            <script>\n            let viz_{stripped_id} = createGridCircuit({self.serialized_circuit}, {moments}, "{self.id}", {self.padding_factor});\n\n            document.getElementById("camera-reset").addEventListener('click', ()  => {{\n            viz_{stripped_id}.scene.setCameraAndControls(viz_{stripped_id}.circuit);\n            }});\n\n            document.getElementById("camera-toggle").addEventListener('click', ()  => {{\n            viz_{stripped_id}.scene.toggleCamera(viz_{stripped_id}.circuit);\n            }});\n            </script>\n        '''

    def get_widget_bundle_name(self) -> str:
        return 'circuit.bundle.js'

    def _serialize_circuit(self) -> str:
        args = []
        moments = self.circuit.moments
        for moment_id, moment in enumerate(moments):
            for item in moment:
                symbol = self._build_3D_symbol(item, moment_id)
                args.append(symbol.to_typescript())
        argument_str = ','.join((str(item) for item in args))
        return f'[{argument_str}]'

    def _build_3D_symbol(self, operation, moment) -> Operation3DSymbol:
        symbol_info = resolve_operation(operation, self._resolvers)
        location_info = []
        for qubit in operation.qubits:
            if isinstance(qubit, cirq.GridQubit):
                location_info.append({'row': qubit.row, 'col': qubit.col})
            elif isinstance(qubit, cirq.LineQubit):
                location_info.append({'row': qubit.x, 'col': 0})
            else:
                raise ValueError('Unsupported qubit type')
        return Operation3DSymbol(symbol_info.labels, location_info, symbol_info.colors, moment)