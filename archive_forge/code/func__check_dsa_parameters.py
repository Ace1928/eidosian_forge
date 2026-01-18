from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
def _check_dsa_parameters(parameters: DSAParameterNumbers) -> None:
    if parameters.p.bit_length() not in [1024, 2048, 3072, 4096]:
        raise ValueError('p must be exactly 1024, 2048, 3072, or 4096 bits long')
    if parameters.q.bit_length() not in [160, 224, 256]:
        raise ValueError('q must be exactly 160, 224, or 256 bits long')
    if not 1 < parameters.g < parameters.p:
        raise ValueError("g, p don't satisfy 1 < g < p.")