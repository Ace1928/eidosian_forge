import dataclasses
import socket
import ssl
import threading
import typing as t
class LDAPResultError(Exception):

    def __init__(self, msg: str, result: 'sansldap.LDAPResult') -> None:
        super().__init__(msg)
        self.result = result

    def __str__(self) -> str:
        inner_msg = super().__str__()
        msg = f'Received LDAPResult error {inner_msg} - {self.result.result_code.name}'
        if self.result.matched_dn:
            msg += f' - Matched DN {self.result.matched_dn}'
        if self.result.diagnostics_message:
            msg += f' - {self.result.diagnostics_message}'
        return msg