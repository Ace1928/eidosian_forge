import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
class QuantumTape(QuantumScript, AnnotatedQueue):
    """A quantum tape recorder, that records and stores variational quantum programs.

    Args:
        ops (Iterable[Operator]): An iterable of the operations to be performed
        measurements (Iterable[MeasurementProcess]): All the measurements to be performed
        prep (Iterable[Operator]): Arguments to specify state preparations to
            perform at the start of the circuit. These should go at the beginning of ``ops``
            instead.

    Keyword Args:
        shots (None, int, Sequence[int], ~.Shots): Number and/or batches of shots for execution.
            Note that this property is still experimental and under development.
        trainable_params (None, Sequence[int]): the indices for which parameters are trainable
        _update=True (bool): Whether or not to set various properties on initialization. Setting
            ``_update=False`` reduces computations if the tape is only an intermediary step.


    **Example**

    Tapes can be constructed by directly providing operations and measurements:

    >>> ops = [qml.BasisState([1,0], wires=0), qml.S(0), qml.T(1)]
    >>> measurements = [qml.state()]
    >>> tape = qml.tape.QuantumTape(ops, measurements)
    >>> tape.circuit
    [BasisState([1, 0], wires=[0]), S(wires=[0]), T(wires=[1]), state(wires=[])]

    They can also be populated into a recording tape via queuing.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.Z(0))

    A ``QuantumTape`` can also be constructed directly from an :class:`~.AnnotatedQueue`:

    .. code-block:: python

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires=0)
            qml.CNOT(wires=[0, 'a'])
            qml.RX(0.133, wires='a')
            qml.expval(qml.Z(0))

        tape = qml.tape.QuantumTape.from_queue(q)

    Once constructed, the tape may act as a quantum circuit and information
    about the quantum circuit can be queried:

    >>> list(tape)
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a']), expval(Z(0))]
    >>> tape.operations
    [RX(0.432, wires=[0]), RY(0.543, wires=[0]), CNOT(wires=[0, 'a']), RX(0.133, wires=['a'])]
    >>> tape.observables
    [expval(Z(0))]
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> tape.wires
    <Wires = [0, 'a']>
    >>> tape.num_params
    3

    The existing circuit is overriden upon exiting a recording context.

    Iterating over the quantum circuit can be done by iterating over the tape
    object:

    >>> for op in tape:
    ...     print(op)
    RX(0.432, wires=[0])
    RY(0.543, wires=[0])
    CNOT(wires=[0, 'a'])
    RX(0.133, wires=['a'])
    expval(Z(0))

    Tapes can also as sequences and support indexing and the ``len`` function:

    >>> tape[0]
    RX(0.432, wires=[0])
    >>> len(tape)
    5

    The :class:`~.CircuitGraph` can also be accessed:

    >>> tape.graph
    <pennylane.circuit_graph.CircuitGraph object at 0x7fcc0433a690>

    Once constructed, the quantum tape can be executed directly on a supported
    device via the :func:`~.pennylane.execute` function:

    >>> dev = qml.device("default.qubit", wires=[0, 'a'])
    >>> qml.execute([tape], dev, gradient_fn=None)
    [array([0.77750694])]

    A new tape can be created by passing new parameters along with the indices
    to be updated to :meth:`~pennylane.tape.QuantumScript.bind_new_parameters`:

    >>> new_tape = tape.bind_new_parameters(params=[0.56], indices=[0])
    >>> tape.get_parameters()
    [0.432, 0.543, 0.133]
    >>> new_tape.get_parameters()
    [0.56, 0.543, 0.133]


    To prevent the tape from being queued use :meth:`~.queuing.QueuingManager.stop_recording`.

    .. code-block:: python

        with qml.tape.QuantumTape() as tape1:
            with qml.QueuingManager.stop_recording():
                with qml.tape.QuantumTape() as tape2:
                    qml.RX(0.123, wires=0)

    Here, tape2 records the RX gate, but tape1 doesn't record tape2.

    >>> tape1.operations
    []
    >>> tape2.operations
    [RX(0.123, wires=[0])]

    This is useful for when you want to transform a tape first before applying it.
    """
    _lock = RLock()
    'threading.RLock: Used to synchronize appending to/popping from global QueueingContext.'

    def __init__(self, ops=None, measurements=None, shots=None, trainable_params=None, _update=True):
        AnnotatedQueue.__init__(self)
        QuantumScript.__init__(self, ops, measurements, shots, trainable_params=trainable_params, _update=_update)

    def __enter__(self):
        QuantumTape._lock.acquire()
        QueuingManager.append(self)
        QueuingManager.add_active_queue(self)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        QueuingManager.remove_active_queue()
        QuantumTape._lock.release()
        self._process_queue()
        self._trainable_params = None

    def adjoint(self):
        adjoint_tape = super().adjoint()
        QueuingManager.append(adjoint_tape)
        return adjoint_tape
    _queue_category = '_ops'

    def _process_queue(self):
        """Process the annotated queue, creating a list of quantum
        operations and measurement processes.

        Sets:
            _ops (list[~.Operation]): Main tape operations
            _measurements (list[~.MeasurementProcess]): Tape measurements

        Also calls `_update()` which sets many attributes.
        """
        self._ops, self._measurements = process_queue(self)
        self._update()

    def __getitem__(self, key):
        """
        Overrides the default because QuantumTape is both a QuantumScript and an AnnotatedQueue.
        If key is an int, the caller is likely indexing the backing QuantumScript. Otherwise, the
        caller is likely indexing the backing AnnotatedQueue.
        """
        if isinstance(key, int):
            return QuantumScript.__getitem__(self, key)
        return AnnotatedQueue.__getitem__(self, key)

    def __setitem__(self, key, val):
        AnnotatedQueue.__setitem__(self, key, val)

    def __hash__(self):
        return QuantumScript.__hash__(self)