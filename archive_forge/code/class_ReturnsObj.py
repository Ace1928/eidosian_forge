import pytest
import cirq
class ReturnsObj:

    def _measurement_key_obj_(self):
        return cirq.MeasurementKey(name='door locker')