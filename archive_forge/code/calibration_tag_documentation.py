from typing import Any, Dict
import cirq
Tag to add onto an Operation that specifies alternate parameters.

    Google devices support the ability to run a procedure from calibration API
    that can tune the device for a specific circuit.  This will return a token
    as part of the result.  Attaching a `CalibrationTag` with that token
    specifies that the gate should use parameters from that specific
    calibration, instead of the default gate parameters.
    