import dataclasses
import socket
import ssl
import threading
import typing as t
def _valid_result(self, result: 'sansldap.LDAPResult', msg: str) -> None:
    if result.result_code != sansldap.LDAPResultCode.SUCCESS:
        raise LDAPResultError(msg, result)