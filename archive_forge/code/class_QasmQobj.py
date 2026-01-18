import copy
import pprint
from types import SimpleNamespace
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.qobj.pulse_qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.qobj.common import QobjDictField, QobjHeader
class QasmQobj:
    """An OpenQASM 2 Qobj."""

    def __init__(self, qobj_id=None, config=None, experiments=None, header=None):
        """Instantiate a new OpenQASM 2 Qobj Object.

        Each OpenQASM 2 Qobj object is used to represent a single payload that will
        be passed to a Qiskit provider. It mirrors the Qobj the published
        `Qobj specification <https://arxiv.org/abs/1809.03452>`_ for OpenQASM
        experiments.

        Args:
            qobj_id (str): An identifier for the qobj
            config (QasmQobjRunConfig): A config for the entire run
            header (QobjHeader): A header for the entire run
            experiments (list): A list of lists of :class:`QasmQobjExperiment`
                objects representing an experiment
        """
        self.header = header or QobjHeader()
        self.config = config or QasmQobjConfig()
        self.experiments = experiments or []
        self.qobj_id = qobj_id
        self.type = 'QASM'
        self.schema_version = '1.3.0'

    def __repr__(self):
        experiments_str = [repr(x) for x in self.experiments]
        experiments_repr = '[' + ', '.join(experiments_str) + ']'
        out = "QasmQobj(qobj_id='{}', config={}, experiments={}, header={})".format(self.qobj_id, repr(self.config), experiments_repr, repr(self.header))
        return out

    def __str__(self):
        out = 'QASM Qobj: %s:\n' % self.qobj_id
        config = pprint.pformat(self.config.to_dict())
        out += 'Config: %s\n' % str(config)
        header = pprint.pformat(self.header.to_dict())
        out += 'Header: %s\n' % str(header)
        out += 'Experiments:\n'
        for experiment in self.experiments:
            out += '%s' % str(experiment)
        return out

    def to_dict(self):
        """Return a dictionary format representation of the OpenQASM 2 Qobj.

        Note this dict is not in the json wire format expected by IBM and Qobj
        specification because complex numbers are still of type complex. Also,
        this may contain native numpy arrays. When serializing this output
        for use with IBM systems, you can leverage a json encoder that converts these
        as expected. For example:

        .. code-block::

            import json
            import numpy

            class QobjEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, numpy.ndarray):
                        return obj.tolist()
                    if isinstance(obj, complex):
                        return (obj.real, obj.imag)
                    return json.JSONEncoder.default(self, obj)

            json.dumps(qobj.to_dict(), cls=QobjEncoder)

        Returns:
            dict: A dictionary representation of the QasmQobj object
        """
        out_dict = {'qobj_id': self.qobj_id, 'header': self.header.to_dict(), 'config': self.config.to_dict(), 'schema_version': self.schema_version, 'type': 'QASM', 'experiments': [x.to_dict() for x in self.experiments]}
        return out_dict

    @classmethod
    def from_dict(cls, data):
        """Create a new QASMQobj object from a dictionary.

        Args:
            data (dict): A dictionary representing the QasmQobj to create. It
                will be in the same format as output by :func:`to_dict`.

        Returns:
            QasmQobj: The QasmQobj from the input dictionary.
        """
        config = None
        if 'config' in data:
            config = QasmQobjConfig.from_dict(data['config'])
        experiments = None
        if 'experiments' in data:
            experiments = [QasmQobjExperiment.from_dict(exp) for exp in data['experiments']]
        header = None
        if 'header' in data:
            header = QobjHeader.from_dict(data['header'])
        return cls(qobj_id=data.get('qobj_id'), config=config, experiments=experiments, header=header)

    def __eq__(self, other):
        if isinstance(other, QasmQobj):
            if self.to_dict() == other.to_dict():
                return True
        return False