from typing import Any, FrozenSet, Mapping, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import value
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class SupportsMeasurementKey(Protocol):
    """An object that is a measurement and has a measurement key or keys.

    Measurement keys are used in referencing the results of a measurement.

    Users are free to implement one of the following. Do not implement multiple
    of these returning different values. The protocol behavior will be
    unexpected in such a case.
    1. `_measurement_key_objs_` returning an iterable of `MeasurementKey`s
    2. `_measurement_key_obj_` returning one `MeasurementKey`
    3. `_measurement_key_names_` returning an iterable of strings
    4. `_measurement_key_name_` returning one string

    Note: Measurements, in contrast to general quantum channels, are
    distinguished by the recording of the quantum operation that occurred.
    That is a general quantum channel may enact the evolution
        $$
        \\rho \\rightarrow \\sum_k A_k \\rho A_k^\\dagger
        $$
    where as a measurement enacts the evolution
        $$
        \\rho \\rightarrow A_k \\rho A_k^\\dagger
        $$
    conditional on the measurement outcome being $k$.
    """

    @doc_private
    def _is_measurement_(self) -> bool:
        """Return if this object is (or contains) a measurement."""

    @doc_private
    def _measurement_key_obj_(self) -> 'cirq.MeasurementKey':
        """Return the key object that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """

    @doc_private
    def _measurement_key_objs_(self) -> Union[FrozenSet['cirq.MeasurementKey'], NotImplementedType, None]:
        """Return the key objects for measurements performed by the receiving object.

        When a measurement occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the measurements
        will be stored.
        """

    @doc_private
    def _measurement_key_name_(self) -> str:
        """Return the string key that will be used to identify this measurement.

        When a measurement occurs, either on hardware, or in a simulation,
        this is the key value under which the results of the measurement
        will be stored.
        """

    @doc_private
    def _measurement_key_names_(self) -> Union[FrozenSet[str], NotImplementedType, None]:
        """Return the string keys for measurements performed by the receiving object.

        When a measurement occurs, either on hardware, or in a simulation,
        these are the key values under which the results of the measurements
        will be stored.
        """

    @doc_private
    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        """Return a copy of this object with the measurement keys remapped.

        This method allows measurement keys to be reassigned at runtime.
        """