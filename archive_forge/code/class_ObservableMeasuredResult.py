import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
@dataclasses.dataclass(frozen=True)
class ObservableMeasuredResult:
    """The result of an observable measurement.

    A list of these is returned by `measure_observables`, or see `flatten_grouped_results` for
    transformation of `measure_grouped_settings` BitstringAccumulators into these objects.

    This is a flattened form of the contents of a `BitstringAccumulator` which may group many
    simultaneously-observable settings into one object. As such, `BitstringAccumulator` has more
    advanced support for covariances between simultaneously-measured observables which is dropped
    when you flatten into these objects.

    Args:
        setting: The setting for which this object contains results
        mean: The mean of the observable specified by `setting`.
        variance: The variance of the observable specified by `setting`.
        repetitions: The number of circuit repetitions used to estimate `setting`.
        circuit_params: The parameters used to resolve the circuit used to prepare the state that
            is being measured.
    """
    setting: InitObsSetting
    mean: float
    variance: float
    repetitions: int
    circuit_params: Mapping[Union[str, sympy.Expr], Union[value.Scalar, sympy.Expr]]

    def __repr__(self):
        return f'cirq.work.ObservableMeasuredResult(setting={self.setting!r}, mean={self.mean!r}, variance={self.variance!r}, repetitions={self.repetitions!r}, circuit_params={self.circuit_params!r})'

    @property
    def init_state(self):
        return self.setting.init_state

    @property
    def observable(self):
        return self.setting.observable

    @property
    def stddev(self):
        return np.sqrt(self.variance)

    def as_dict(self) -> Dict[str, Any]:
        """Return the contents of this class as a dictionary.

        This makes records suitable for construction of a Pandas dataframe. The circuit parameters
        are flattened into the top-level of this dictionary.
        """
        record = dataclasses.asdict(self)
        del record['circuit_params']
        del record['setting']
        record['init_state'] = self.init_state
        record['observable'] = self.observable
        circuit_param_dict = {f'param.{k}': v for k, v in self.circuit_params.items()}
        record.update(**circuit_param_dict)
        return record

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)