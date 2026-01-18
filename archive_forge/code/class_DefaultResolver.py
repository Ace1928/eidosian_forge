import abc
import dataclasses
from typing import Iterable, List, Optional
import cirq
from cirq.protocols.circuit_diagram_info_protocol import CircuitDiagramInfoArgs
class DefaultResolver(SymbolResolver):
    """Default symbol resolver implementation. Takes information
    from circuit_diagram_info, if unavailable, returns information representing
    an unknown symbol.
    """
    _SYMBOL_COLORS = {'@': 'black', 'H': 'yellow', 'I': 'orange', 'X': 'black', 'Y': 'pink', 'Z': 'cyan', 'S': '#90EE90', 'T': '#CBC3E3'}

    def resolve(self, operation: cirq.Operation) -> Optional[SymbolInfo]:
        """Checks for the _circuit_diagram_info attribute of the operation,
        and if it exists, build the symbol information from it. Otherwise,
        builds symbol info for an unknown operation.

        Args:
            operation: the cirq.Operation object to resolve
        """
        try:
            info = cirq.circuit_diagram_info(operation)
        except TypeError:
            return SymbolInfo.unknown_operation(cirq.num_qubits(operation))
        wire_symbols = info.wire_symbols
        symbol_exponent = info._wire_symbols_including_formatted_exponent(CircuitDiagramInfoArgs.UNINFORMED_DEFAULT)
        symbol_info = SymbolInfo(list(symbol_exponent), [])
        for symbol in wire_symbols:
            symbol_info.colors.append(DefaultResolver._SYMBOL_COLORS.get(symbol, 'gray'))
        return symbol_info