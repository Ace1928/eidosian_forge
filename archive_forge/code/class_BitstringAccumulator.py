import dataclasses
import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq._compat import proper_repr
from cirq.work.observable_settings import (
class BitstringAccumulator:
    """A mutable container of bitstrings and associated metadata populated
    during a `measure_observables` run.

    This object contains all raw results and can be serialized via JSON to
    keep a record of your experiment results. There are also various
    utility methods that can be used to chain a series of BitstringAccumulator
    results into a form more suitable for analysis like a pandas DataFrame.

    By default, this will be initialized empty. This should only be mutated
    by calling `consume_results`. Do not mutate values directly.

    Args:
        meas_spec: The specification with the particular run used to
            gather these bitstrings. There should be a 1-to-1 correspondence
            between bitstring accumulators and circuits run on a quantum
            sampler.
        simul_settings: The list of settings consistent with this
            measurement spec, usually the result of grouping a list
            of requested settings. This list need not be exhausted:
            any setting consistent with the `meas_spec` can be queried
            with methods that take a setting as argument (e.g. `mean`,
            `variance`) whether or not they are provided up-front in
            `simul_settings`. Those methods that do *not* take a setting
            as an argument (e.g. `means`, `variances`) will report all values
            for the settings in `simul_settings`.
        qubit_to_index: A mapping from qubits to contiguous indices starting
            from zero. This allows us to store bitstrings as a 2d numpy array.
        bitstrings: The bitstrings to record.
        chunksizes: This class accumulates bitstrings from potentially several
            "chunked" processor runs. Each chunk has a certain number of
            repetitions, recorded in this array. This theoretically
            allows you to re-split up the bitstring array should the need
            arise. The total number of repetitions is the sum of this 1d array.
        timestamps: We record a timestamp for each request/chunk. This
            1d array will have the same length as `chunksizes`.
        readout_calibration: The result of `calibrate_readout_error`. When requesting
            means and variances, if this is not `None`, we will use the
            calibrated value to correct the requested quantity. This is a
            `BitstringAccumulator` containing the results of measuring Z
            observables with readout symmetrization enabled. This class
            does *not* validate that both this parameter and the
            `BitstringAccumulator` under construction contain measurements taken
            with readout symmetrization turned on.
    """

    def __init__(self, meas_spec: _MeasurementSpec, simul_settings: List[InitObsSetting], qubit_to_index: Dict['cirq.Qid', int], bitstrings: Optional[np.ndarray]=None, chunksizes: Optional[np.ndarray]=None, timestamps: Optional[np.ndarray]=None, readout_calibration: Optional['BitstringAccumulator']=None):
        self._meas_spec = meas_spec
        self._simul_settings = simul_settings
        self._qubit_to_index = qubit_to_index
        self._readout_calibration = readout_calibration
        if bitstrings is None:
            n_bits = len(qubit_to_index)
            self.bitstrings = np.zeros((0, n_bits), dtype=np.uint8)
        else:
            self.bitstrings = np.asarray(bitstrings, dtype=np.uint8)
        if chunksizes is None:
            self.chunksizes = np.zeros((0,), dtype=np.int64)
        else:
            self.chunksizes = np.asarray(chunksizes, dtype=np.int64)
        if timestamps is None:
            self.timestamps = np.zeros((0,), dtype='datetime64[us]')
        else:
            self.timestamps = np.asarray(timestamps, dtype='datetime64[us]')
        if len(self.chunksizes) != len(self.timestamps):
            raise ValueError('Invalid BitstringAccumulator state. `chunksizes` and `timestamps` must have the same length.')
        if np.sum(self.chunksizes) != len(self.bitstrings):
            raise ValueError('Invalid BitstringAccumulator state. `chunksizes` must sum to the number of bitstrings.')

    @property
    def meas_spec(self):
        return self._meas_spec

    @property
    def max_setting(self):
        return self.meas_spec.max_setting

    @property
    def circuit_params(self):
        return self.meas_spec.circuit_params

    @property
    def simul_settings(self):
        return self._simul_settings

    @property
    def qubit_to_index(self):
        return self._qubit_to_index

    def consume_results(self, bitstrings):
        """Add bitstrings sampled according to `meas_spec`.

        We don't validate that bitstrings were sampled correctly according
        to `meas_spec` (how could we?) so please be careful. Consider
        using `measure_observables` rather than calling this method yourself.
        """
        if bitstrings.dtype != np.uint8:
            raise ValueError('`bitstrings` should be of type np.uint8')
        self.bitstrings = np.append(self.bitstrings, bitstrings, axis=0)
        self.chunksizes = np.append(self.chunksizes, [len(bitstrings)], axis=0)
        self.timestamps = np.append(self.timestamps, [np.datetime64(datetime.datetime.now())])

    @property
    def n_repetitions(self):
        return len(self.bitstrings)

    @property
    def results(self) -> Iterable[ObservableMeasuredResult]:
        """Yield individual setting results as `ObservableMeasuredResult`
        objects."""
        for setting in self._simul_settings:
            yield ObservableMeasuredResult(setting=setting, mean=self.mean(setting), variance=self.variance(setting), repetitions=len(self.bitstrings), circuit_params=self._meas_spec.circuit_params)

    @property
    def records(self):
        """Yield individual setting results as dictionary records.

        This is suitable for passing to pd.DataFrame constructor, perhaps
        after chaining these results with those from other BitstringAccumulators.
        """
        for result in self.results:
            yield result.as_dict()

    def _json_dict_(self):
        from cirq.study.result import _pack_digits

        def ndarray_to_hex_str(a):
            return _pack_digits(a, pack_bits='never')[0]
        return {'meas_spec': self.meas_spec, 'simul_settings': self.simul_settings, 'qubit_to_index': list(self.qubit_to_index.items()), 'bitstrings': ndarray_to_hex_str(self.bitstrings), 'chunksizes': ndarray_to_hex_str(self.chunksizes), 'timestamps': ndarray_to_hex_str(self.timestamps)}

    @classmethod
    def _from_json_dict_(cls, *, meas_spec, simul_settings, qubit_to_index, bitstrings, chunksizes, timestamps, **kwargs):
        from cirq.study.result import _unpack_digits

        def hex_str_to_ndarray(hexstr):
            return _unpack_digits(hexstr, binary=False, dtype=None, shape=None)
        return cls(meas_spec=meas_spec, simul_settings=simul_settings, qubit_to_index=dict(qubit_to_index), bitstrings=hex_str_to_ndarray(bitstrings), chunksizes=hex_str_to_ndarray(chunksizes), timestamps=hex_str_to_ndarray(timestamps))

    def __eq__(self, other):
        if not isinstance(other, BitstringAccumulator):
            return NotImplemented
        if self.max_setting != other.max_setting or self.simul_settings != other.simul_settings or self.circuit_params != other.circuit_params or (self.qubit_to_index != other.qubit_to_index):
            return False
        if not np.array_equal(self.bitstrings, other.bitstrings):
            return False
        if not np.array_equal(self.chunksizes, other.chunksizes):
            return False
        if not np.array_equal(self.timestamps, other.timestamps):
            return False
        return True

    def summary_string(self, setting: InitObsSetting, number_fmt='.3f'):
        return f'{setting}: {self.mean(setting):{number_fmt}} +- {self.stderr(setting):{number_fmt}}'

    def __repr__(self):
        return f'cirq.work.BitstringAccumulator(meas_spec={self.meas_spec!r}, simul_settings={self.simul_settings!r}, qubit_to_index={self.qubit_to_index!r}, bitstrings={proper_repr(self.bitstrings)}, chunksizes={proper_repr(self.chunksizes)}, timestamps={proper_repr(self.timestamps)}, readout_calibration={self._readout_calibration!r})'

    def __str__(self):
        s = f'Accumulator {self.max_setting}; {self.n_repetitions} repetitions\n'
        s += '\n'.join(('  ' + self.summary_string(setting) for setting in self._simul_settings))
        return s

    def covariance(self, *, atol=1e-08) -> np.ndarray:
        """Compute the covariance matrix for the estimators of all settings.

        Like `variance`, this is the covariance of the sampling distribution
        of the sample mean. Practically, it is the 'normal' covariance
        divided by the number of observations (bitstrings).

        Args:
            atol: The absolute tolerance for asserting coefficients are real.

        Raises:
            ValueError: If there are no measurements.
        """
        if len(self.bitstrings) == 0:
            raise ValueError('No measurements')
        all_obs_vals = np.array([_obs_vals_from_measurements(bitstrings=self.bitstrings, qubit_to_index=self._qubit_to_index, observable=setting.observable, atol=atol) for setting in self._simul_settings])
        if all_obs_vals.shape[0] == 1:
            cov = np.array([[np.var(all_obs_vals[0], ddof=1)]])
            return cov
        cov = np.cov(all_obs_vals, ddof=1) / all_obs_vals.shape[1]
        return cov

    def _validate_setting(self, setting: InitObsSetting, what: str):
        mws = _max_weight_state([self.max_setting.init_state, setting.init_state])
        mwo = _max_weight_observable([self.max_setting.observable, setting.observable])
        if mws is None or mwo is None:
            raise ValueError(f"You requested the {what} for a setting that is not compatible with this BitstringAccumulator's meas_spec.")

    def variance(self, setting: InitObsSetting, *, atol: float=1e-08):
        """Compute the variance of the estimators of the given setting.

        This is the normal variance divided by the number of samples to estimate
        the certainty of our estimate of the mean. It is the standard error
        of the mean, squared.

        This uses `ddof=1` during the call to `np.var` for an unbiased estimator
        of the variance in a hypothetical infinite population for consistency
        with `BitstringAccumulator.covariance()` but differs from the default
        for `np.var`.

        Args:
            setting: The initial state and observable.
            atol: The absolute tolerance for asserting coefficients are real.

        Raises:
            ValueError: If there were no measurements.
        """
        if len(self.bitstrings) == 0:
            raise ValueError('No measurements')
        self._validate_setting(setting, what='variance')
        mean, var = _stats_from_measurements(bitstrings=self.bitstrings, qubit_to_index=self._qubit_to_index, observable=setting.observable, atol=atol)
        if self._readout_calibration is not None:
            a = mean
            if np.isclose(a, 0, atol=atol):
                return np.inf
            var_a = var
            ro_setting = _setting_to_z_observable(setting)
            b = self._readout_calibration.mean(ro_setting)
            if np.isclose(b, 0, atol=atol):
                return np.inf
            var_b = self._readout_calibration.variance(ro_setting)
            f = a / b
            var = f ** 2 * (var_a / a ** 2 + var_b / b ** 2)
        return var

    def stderr(self, setting: InitObsSetting, *, atol: float=1e-08):
        """The standard error of the estimators for `setting`."""
        return np.sqrt(self.variance(setting, atol=atol))

    def means(self, *, atol: float=1e-08) -> np.ndarray:
        """Estimates of the means of the settings in this accumulator."""
        return np.asarray([self.mean(setting, atol=atol) for setting in self.simul_settings])

    def mean(self, setting: InitObsSetting, *, atol: float=1e-08):
        """Estimates of the mean of `setting`."""
        if len(self.bitstrings) == 0:
            raise ValueError('No measurements')
        self._validate_setting(setting, what='mean')
        mean, _ = _stats_from_measurements(bitstrings=self.bitstrings, qubit_to_index=self._qubit_to_index, observable=setting.observable, atol=atol)
        if self._readout_calibration is not None:
            ro_setting = _setting_to_z_observable(setting)
            return mean / self._readout_calibration.mean(ro_setting, atol=atol)
        return mean