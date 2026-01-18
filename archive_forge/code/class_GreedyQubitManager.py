import cirq
@cirq._compat.deprecated_class(deadline='v1.4', fix='Use cirq.GreedyQubitManager instead.')
class GreedyQubitManager(cirq.GreedyQubitManager):
    pass